import os
import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS

@HOOKS.register_module()
class SaveTeacherHook(Hook):
    """
    Save teacher weights after each validation epoch alongside student checkpoints.
    Works with SemiMeanTeacher model.
    """
    def __init__(self, enabled: bool = True, file_prefix: str = 'teacher_epoch'):
        self.enabled = enabled
        self.file_prefix = file_prefix

    def after_val_epoch(self, runner, metrics=None):
        if not self.enabled:
            return
        model = runner.model
        if not hasattr(model, 'state_dict_teacher'):
            return
        ckpt = model.state_dict_teacher()
        path = os.path.join(runner.work_dir, f'{self.file_prefix}_{runner.epoch}.pth')
        torch.save({'state_dict': ckpt}, path)
        runner.logger.info(f'[SaveTeacherHook] Saved teacher checkpoint to: {path}')
