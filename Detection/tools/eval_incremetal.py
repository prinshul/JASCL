import argparse
import torch
import os
import sys
from mmengine.config import Config
from mmengine.runner import Runner
from datetime import datetime

def run_eval(cfg_path, checkpoint):
    print(f" Running evaluation for: {cfg_path}")
    print(f"   Using checkpoint: {checkpoint}\n")

    cfg = Config.fromfile(cfg_path)
    cfg.load_from = checkpoint

    # force test mode
    cfg.test_evaluator = cfg.get('test_evaluator', cfg.val_evaluator)
    cfg.test_dataloader = cfg.get('test_dataloader', cfg.val_dataloader)

    runner = Runner.from_cfg(cfg)
    runner.test()

def main():
    parser = argparse.ArgumentParser(description="Incremental test runner")
    parser.add_argument(
        "--base", type=str, default="configs/soft_incremental/base.py",
        help="Config path for base step"
    )
    parser.add_argument(
        "--base_ckpt", type=str, default="work_dirs/ssl_coco_o_base/best_coco_bbox_mAP_epoch_100.pth",
        help="Checkpoint for base step"
    )
    parser.add_argument(
        "--step1", type=str, default="configs/soft_incremental/step1.py",
        help="Config path for step1"
    )
    parser.add_argument(
        "--step1_ckpt", type=str, default="work_dirs/ssl_coco_o_step1/best_teacher_coco_bbox_mAP_epoch_0.pth",
        help="Checkpoint for step1"
    )
    parser.add_argument(
        "--step2", type=str, default="configs/soft_incremental/step2.py",
        help="Config path for step2"
    )
    parser.add_argument(
        "--step2_ckpt", type=str, default="work_dirs/ssl_coco_o_step2/best_teacher_coco_bbox_mAP_epoch_0.pth",
        help="Checkpoint for step2"
    )
    args = parser.parse_args()
    
    now_time = datetime.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    # print("Base Testing - Start time :", s1)
    # Run evaluations in sequence
    # run_eval(args.base, args.base_ckpt)
    
    now_time = datetime.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("Step1 Testing - Start time :", s1)
    run_eval(args.step1, args.step1_ckpt)
    
    # now_time = datetime.now()
    # s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # # mm/dd/YY H:M:S format
    # print("Step2 Testing - Start time :", s1)
    # run_eval(args.step2, args.step2_ckpt)


if __name__ == "__main__":
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
