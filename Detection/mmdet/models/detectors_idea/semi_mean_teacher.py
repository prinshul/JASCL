# mmdet/models/detectors/semi_mean_teacher.py
from copy import deepcopy
from typing import Dict, Any, List
import torch
import torch.nn.functional as F
from mmengine.model import BaseModel
from mmdet.registry import MODELS
from mmdet.structures import DetDataSample
from mmengine.structures import InstanceData


@MODELS.register_module()
class SemiMeanTeacher(BaseModel):
    """
    SemiMeanTeacher wrapper. Build student detector from 'student_cfg' nested dict in config.
    Compute supervised loss on sup, pseudo-supervised loss on unsup using teacher pseudo labels
    (threshold + top-k) and a feature-level consistency loss. Update via optim_wrapper.update_params.
    """

    def __init__(self,
                 student_cfg: Dict[str, Any],
                 ema_decay: float = 0.999,
                 pseudo_label: Dict[str, Any] = None,
                 loss_weights: Dict[str, float] = None,
                 data_preprocessor: Dict[str, Any] = None,
                 init_cfg: Dict[str, Any] = None,
                 **kwargs):
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        # Build student detector (FasterRCNN etc.)
        if student_cfg is None:
            raise ValueError("student_cfg must be provided in model block")
        self.student = MODELS.build(student_cfg)

        # Teacher is EMA copy
        self.teacher = deepcopy(self.student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

        self.ema_decay = float(ema_decay)
        self.pseudo_cfg = dict(score_thr=0.5, max_per_img=200)
        if pseudo_label:
            self.pseudo_cfg.update(pseudo_label)

        self.loss_w = dict(sup=1.0, unsup=1.0, cons=0.1)
        if loss_weights:
            self.loss_w.update(loss_weights)

    def forward(self, *args, mode: str = 'loss', **kwargs):
        # make the model instantiable for MMEngine
        if mode == 'loss':
            return self.student.forward(*args, mode='loss', **kwargs)
        if mode == 'predict':
            return self.student.predict(*args, **kwargs)
        if mode == 'tensor':
            return self.student.forward(*args, mode='tensor', **kwargs)
        raise ValueError(f"Invalid forward mode: {mode}")

    @torch.no_grad()
    def _ensure_teacher_device(self, device: torch.device):
        # move teacher if required
        cur = None
        for p in self.teacher.parameters():
            cur = p.device
            break
        if cur != device:
            self.teacher.to(device)

    @torch.no_grad()
    def _filter_teacher_preds(self, teacher_preds: List[DetDataSample]) -> List[InstanceData]:
        thr = float(self.pseudo_cfg.get('score_thr', 0.5))
        max_per = int(self.pseudo_cfg.get('max_per_img', 200))
        output = []
        for ds in teacher_preds:
            inst = ds.pred_instances
            if inst.bboxes.numel() == 0:
                output.append(InstanceData(bboxes=torch.zeros((0, 4)), labels=torch.zeros((0,), dtype=torch.long),
                                           scores=torch.zeros((0,))))
                continue
            scores = inst.scores
            keep = scores >= thr
            if keep.sum() == 0:
                output.append(InstanceData(bboxes=torch.zeros((0, 4)), labels=torch.zeros((0,), dtype=torch.long),
                                           scores=torch.zeros((0,))))
                continue
            bboxes = inst.bboxes[keep]
            labels = inst.labels[keep]
            scores = inst.scores[keep]
            if bboxes.shape[0] > max_per:
                topk = torch.topk(scores, max_per).indices
                bboxes = bboxes[topk]
                labels = labels[topk]
                scores = scores[topk]
            inst_out = InstanceData()
            inst_out.bboxes = bboxes
            inst_out.labels = labels
            inst_out.scores = scores
            output.append(inst_out)
        return output

    def _pseudo_to_datasamples(self, teacher_filtered: List[InstanceData], metas: List[DetDataSample]) -> List[DetDataSample]:
        outs: List[DetDataSample] = []
        for inst, meta in zip(teacher_filtered, metas):
            ds = DetDataSample()
            ds.set_metainfo(meta.metainfo)
            ds.gt_instances = InstanceData()
            if inst.bboxes.numel() == 0:
                ds.gt_instances.bboxes = torch.zeros((0, 4), device=inst.bboxes.device)
                ds.gt_instances.labels = torch.zeros((0,), dtype=torch.long, device=inst.labels.device)
            else:
                ds.gt_instances.bboxes = inst.bboxes
                ds.gt_instances.labels = inst.labels
            outs.append(ds)
        return outs

    def _feature_consistency(self, inputs_unsup: torch.Tensor) -> torch.Tensor:
        # compare student and teacher feature maps (MSE)
        stu_feats = self.student.extract_feat(inputs_unsup)
        with torch.no_grad():
            tea_feats = self.teacher.extract_feat(inputs_unsup)
        total = torch.tensor(0.0, device=stu_feats[0].device)
        cnt = 0
        for sf, tf in zip(stu_feats, tea_feats):
            total = total + F.mse_loss(sf, tf.detach())
            cnt += 1
        if cnt == 0:
            return total
        return total / cnt

    def train_step(self, data: Dict[str, Any], optim_wrapper):
        # data: {'sup': {'inputs','data_samples'}, 'unsup': {'inputs','data_samples'}}
        sup = data.get('sup', None)
        if sup is None:
            raise RuntimeError("SemiMeanTeacher expects 'sup' key in training batch.")

        inputs_sup = sup['inputs']
        samples_sup = sup['data_samples']

        # supervised loss from student
        sup_losses = self.student.forward(inputs_sup, samples_sup, mode='loss')
        sup_loss, sup_log_vars = self.parse_losses(sup_losses)
        total_loss = self.loss_w['sup'] * sup_loss
        log_vars = {f"sup_{k}": v for k, v in sup_log_vars.items()}

        # unsup branch
        unsup = data.get('unsup', None)
        if unsup is not None:
            inputs_unsup = unsup['inputs']
            samples_unsup = unsup.get('data_samples', None)

            device = next(self.student.parameters()).device
            self._ensure_teacher_device(device)

            # teacher predictions -> filter
            with torch.no_grad():
                teacher_preds = self.teacher.predict(inputs_unsup, samples_unsup)
            teacher_filtered = self._filter_teacher_preds(teacher_preds)

            # pseudo labels into DetDataSample list
            pseudo_ds = self._pseudo_to_datasamples(teacher_filtered, samples_unsup)
            any_pseudo = any([d.gt_instances.bboxes.numel() > 0 for d in pseudo_ds])

            if any_pseudo:
                unsup_losses = self.student.forward(inputs_unsup, pseudo_ds, mode='loss')
                unsup_loss, unsup_log_vars = self.parse_losses(unsup_losses)
                total_loss = total_loss + self.loss_w['unsup'] * unsup_loss
                log_vars.update({f"unsup_{k}": v for k, v in unsup_log_vars.items()})

            # feature consistency loss
            cons_loss = self._feature_consistency(inputs_unsup)
            total_loss = total_loss + self.loss_w['cons'] * cons_loss
            log_vars.update({"cons_loss": float(cons_loss.item() if torch.is_tensor(cons_loss) else cons_loss)})

        # optimizer step via optim_wrapper
        optim_wrapper.update_params(total_loss)

        # EMA update teacher
        with torch.no_grad():
            m = self.ema_decay
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)

        num_samples = len(samples_sup) if hasattr(samples_sup, '__len__') else 0
        return dict(log_vars=log_vars, num_samples=num_samples)

    def predict(self, batch_inputs, batch_data_samples=None, **kwargs):
        return self.student.predict(batch_inputs, batch_data_samples, **kwargs)

    def state_dict_teacher(self):
        return {k: v.detach().cpu() for k, v in self.teacher.state_dict().items()}
