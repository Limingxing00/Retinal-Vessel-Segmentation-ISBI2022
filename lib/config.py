# -*- coding: utf-8 -*-
# @Time    : 2019/11/18 21:46
# @Author  : Mingxing Li
# @FileName: config.py
# @Software: PyCharm

import argparse
import yaml


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='My setting, Mingxing')

    parser.add_argument(
        '--cfg', dest='cfg_file',
        help='Config file for training (and optionally testing)')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=10, type=int)

    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', default=False, action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.
    parser.add_argument(
        '--bs', dest='batch_size',
        help='Explicitly specify to overwrite the value comed from cfg_file.',
        type=int)

    parser.add_argument(
        '--iter_size',
        help='Update once every iter_size steps, as in Caffe.',
        default=1, type=int)

    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)

    # Epoch
    # parser.add_argument(
    #     '--start_step',
    #     help='Starting step count for training epoch. 0-indexed.',
    #     default=0, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        default=True, action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    f = open(args.cfg_file)
    d = yaml.load(f)
    print(d)