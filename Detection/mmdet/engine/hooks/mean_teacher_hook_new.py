import copy
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MeanTeacherHook_new(Hook):
    """Mean Teacher Hook for Semi-supervised Object Detection (Incremental).
    Expects each batch from SemiDataset as dict(sup=..., unsup=...).
    """

    def __init__(self, ema_decay=0.999, unsup_weight=1.0):
        self.ema_decay = ema_decay
        self.unsup_weight = unsup_weight
        self.teacher_model = None

    def before_train(self, runner):
        # Build teacher as EMA copy of student
        student = runner.model
        self.teacher_model = copy.deepcopy(student)
        self.teacher_model.eval()
        for p in self.teacher_model.parameters():
            p.requires_grad = False
        runner.logger.info("MeanTeacher teacher model created.")

    def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
        """Compute unsupervised loss and EMA update."""
        # Student supervised loss already computed by MMDet
        if data_batch is None:
            return

        if "unsup" not in data_batch:
            return  # base step (no semi)

        student = runner.model
        teacher = self.teacher_model

        # Move unsup batch to device
        device = next(student.parameters()).device
        unsup_data = data_batch["unsup"]
        for k in unsup_data:
            if torch.is_tensor(unsup_data[k]):
                unsup_data[k] = unsup_data[k].to(device)

        with torch.no_grad():
            teacher_out = teacher(**unsup_data)

        student_out = student(**unsup_data)

        # --- consistency loss ---
        loss_unsup = 0.0
        for t_out, s_out in zip(teacher_out, student_out):
            if "cls_scores" in s_out and "cls_scores" in t_out:
                loss_unsup += torch.nn.functional.mse_loss(
                    s_out["cls_scores"], t_out["cls_scores"]
                )
            if "bbox_preds" in s_out and "bbox_preds" in t_out:
                loss_unsup += torch.nn.functional.mse_loss(
                    s_out["bbox_preds"], t_out["bbox_preds"]
                )

        loss_unsup = self.unsup_weight * loss_unsup

        # Backprop unsupervised loss
        runner.optimizer.zero_grad()
        loss_unsup.backward()
        runner.optimizer.step()

        # EMA update of teacher
        self._update_ema(student, teacher)

        runner.log_buffer.update({"loss_unsup": float(loss_unsup)})

    def _update_ema(self, student, teacher):
        with torch.no_grad():
            for s_param, t_param in zip(student.parameters(), teacher.parameters()):
                t_param.data.mul_(self.ema_decay).add_(
                    s_param.data, alpha=1 - self.ema_decay
                )
