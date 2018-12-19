""" Training Script """
import argparse
import distutils.util
import os
import sys
import pickle
import resource
import traceback
import logging
from collections import defaultdict

import numpy as np
import yaml
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
sys.path.append('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron/lib')
sys.path.append('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron/lib_ss')

import nn as mynn
import utils.net as net_utils
import utils.misc as misc_utils

from utils.detectron_weight_helper import load_detectron_weight
from utils.logging import log_stats
from utils.timer import Timer
from utils.training_stats import TrainingStats

from ss_core.ss_config import cfg as cfg
from ss_dataloader.ss_loader import SemanticSegmentationDataset, Mode

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RuntimeError: received 0 items of ancdata. Issue: pytorch/pytorch#973
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Train a semantic segmentation network')

    parser.add_argument(
        '--txtfile', dest='txtfile', required=True,
        help='txtfile that contains the useful data paths')

    parser.add_argument(
        '--mode',dest='mode', required=True,
        help='train, val or test'
        )

    #parser.add_argument(
     #   '--cfg', dest='cfg_file', required=True,
      #  help='Config file)')

    parser.add_argument(
        '--disp_interval',
        help='Display training info every N iterations',
        default=100, type=int)
    parser.add_argument(
        '--no_cuda', dest='cuda', help='Do not use CUDA device', action='store_false')

    # Optimization
    # These options has the highest prioity and can overwrite the values in config file
    # or values set by set_cfgs. `None` means do not overwrite.

    parser.add_argument(
        '--nw', dest='num_workers',
        help='Explicitly specify to overwrite number of workers to load data. Defaults to 4',
        type=int)

    parser.add_argument(
        '--o', dest='optimizer', help='Training optimizer.',
        default=None)
    parser.add_argument(
        '--lr', help='Base learning rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_gamma',
        help='Learning rate decay rate.',
        default=None, type=float)
    parser.add_argument(
        '--lr_decay_epochs',
        help='Epochs to decay the learning rate on. '
             'Decay happens on the beginning of a epoch. '
             'Epoch is 0-indexed.',
        default=[4, 5], nargs='+', type=int)

    # Epoch
    parser.add_argument(
        '--start_iter',
        help='Starting iteration for first training epoch. 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--start_epoch',
        help='Starting epoch count. Epoch is 0-indexed.',
        default=0, type=int)
    parser.add_argument(
        '--epochs', dest='num_epochs',
        help='Number of epochs to train',
        default=10, type=int)

    # Resume training: requires same iterations per epoch
    parser.add_argument(
        '--resume',
        help='resume to training on a checkpoint',
        action='store_true')

    parser.add_argument(
        '--no_save', help='do not save anything', action='store_true')

    parser.add_argument(
        '--ckpt_num_per_epoch',
        help='number of checkpoints to save in each epoch. '
             'Not include the one at the end of an epoch.',
        default=3, type=int)

    parser.add_argument(
        '--load_ckpt', help='checkpoint path to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--use_tfboard', help='Use tensorflow tensorboard to log training info',
        action='store_true')

    return parser.parse_args()


if __name__=='__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    if args.cuda or cfg.NUM_GPUS > 0:
        cfg.CUDA = True
    else:
        raise ValueError("Need Cuda device to run !")

    txtfile = args.txtfile
    mode_str = args.mode.lower()
    #'val' is the same with 'val_no_pred'
    if not mode_str in ['train','val','val_pred','val_no_pred','test']:
        raise ValueError('Unexpected mode: {}').format(args.mode_str)

    mode = Mode.TRAIN
    if mode_str == 'train':
        mode = Mode.TRAIN
    elif mode_str == 'val' or mode_str == 'val_no_pred':
        mode = Mode.VAL_NO_PRED
    elif mode_str == 'val_pred':
        mode = Mode.VAL_PRED
    elif mode_str == 'test':
        mode = Mode.TEST
    else:
        raise Exception('Mode is not valid.')
    #Here can insert multi-GPU training config changes.
    #Not implemented yet
    #

    use_data_augmentation = False
    if mode == Mode.TRAIN:
        use_data_augmentation = True
        dataset = SemanticSegmentationDataset(txtfile, mode, use_data_augmentation, assertNum = None)
        dataloader = DataLoader(dataset, batch_size = 1,shuffle = True)
        dataiterator = iter(dataloader)
        input_data = next(dataiterator)
        #reshape data to 4 dimentions
        #input_data[0]: img 
        #input_data[1]: gt
        #input_data[2]: index
        input_data[0] = input_data[0].view(-1,*input_data[0].shape[2:])
        input_data[1] = input_data[1].view(-1,*input_data[1].shape[2:])
        

    else:
        dataloader = DataLoader(dataset, batch_size = 1,shuffle = False)







