import sys
# sys.path.append("mmgs/gaussian-splatting")

import argparse
import copy
import os
import mmcv
from mmcv import Config, DictAction
from mmgs.utils import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='arguments in dict')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    
    visualizer = Visualizer(cfg)
    visualizer.run()

if __name__ == '__main__':
    main()