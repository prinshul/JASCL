# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import sys, torch
from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from datetime import datetime
from mmdet.utils import setup_cache_size_limit_of_dynamo
import sys
import importlib
# Add this hook class to your training script (train.py) or create a separate hooks file
modules_to_reload = [
    'mmdet.models.detectors.soft_teacher',  # or wherever your SoftTeacher is
    'mmdet.models.roi_heads.bbox_heads.convfc_bbox_head',  # if you modified bbox head
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])

class StochasticGradientUpdateHook:
    """Hook to update gradient information in stochastic classifiers."""
    
    def __init__(self):
        self.priority = 'NORMAL'
    
    def after_train_iter(self, runner):
        """Update gradient information after each training iteration."""
        model = runner.model
        
        # For semi-supervised models (SoftTeacher)
        if hasattr(model, 'student') and hasattr(model.student, 'roi_head'):
            bbox_head = model.student.roi_head.bbox_head
            if hasattr(bbox_head, 'update_gradient_info'):
                bbox_head.update_gradient_info()
        
        # For teacher model if it exists
        if hasattr(model, 'teacher') and hasattr(model.teacher, 'roi_head'):
            bbox_head = model.teacher.roi_head.bbox_head
            if hasattr(bbox_head, 'update_gradient_info'):
                bbox_head.update_gradient_info()
        
        # For single model (non-semi-supervised)
        elif hasattr(model, 'roi_head'):
            bbox_head = model.roi_head.bbox_head
            if hasattr(bbox_head, 'update_gradient_info'):
                bbox_head.update_gradient_info()
def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--auto-scale-lr',
        action='store_true',
        help='enable automatically scaling LR.')
    parser.add_argument(
        '--resume',
        nargs='?',
        type=str,
        const='auto',
        help='If specify checkpoint path, resume from it, while if not '
        'specify, try to auto resume from the latest checkpoint '
        'in the work directory.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    setup_cache_size_limit_of_dynamo()

    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    
    if args.amp is True:
        cfg.optim_wrapper.type = 'AmpOptimWrapper'
        cfg.optim_wrapper.loss_scale = 'dynamic'

    
    if args.auto_scale_lr:
        if 'auto_scale_lr' in cfg and \
                'enable' in cfg.auto_scale_lr and \
                'base_batch_size' in cfg.auto_scale_lr:
            cfg.auto_scale_lr.enable = True
        else:
            raise RuntimeError('Can not find "auto_scale_lr" or '
                               '"auto_scale_lr.enable" or '
                               '"auto_scale_lr.base_batch_size" in your'
                               ' configuration file.')

    
    if args.resume == 'auto':
        cfg.resume = True
        cfg.load_from = None
    elif args.resume is not None:
        cfg.resume = True
        cfg.load_from = args.resume

    
    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    sys.stdout.flush()
    runner.train()


if __name__ == '__main__':
    
    # current date and time
    now_time = datetime.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("Start time :", s1)
    
    print("\nGPU Details : ")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    print("\n",os.getcwd(),"\n")
    torch.cuda.empty_cache()
    sys.stdout.flush()

    main()
    
    now_time = datetime.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("End time :", s1)