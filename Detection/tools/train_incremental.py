import argparse
from mmengine.runner import Runner
from mmengine import Config


def parse_args():
    parser = argparse.ArgumentParser(description='Incremental Semi-Supervised Training')
    parser.add_argument('config', help='config file')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir

    runner = Runner.from_cfg(cfg)

    if args.resume:
        runner.resume()
    else:
        runner.train()


if __name__ == '__main__':
    main()
