from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2

import torch
import torch.nn as nn
from torch.autograd import Variable
def add_path(path):
    if path not in sys.path:
        sys.path.append(path)

add_path('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron')
add_path('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/VOSDetectron/lib')

import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from lib_vos.vos_modeling.generalized_rcnn_predictor_with_boxes import Generalized_RCNN_Predictor_with_Boxes
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
from imdb.vos import davis_db
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')

    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args


def main():
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    args = parse_args()
    print('Called with args:')
    print(args)
    train_db = davis_db.DAVIS_imdb(split = 'train')
    train_db.set_to_sequence(0)
    im = train_db.get_image_cv2(0)
    boxes = train_db.get_bboxes(0)
    boxes = np.array(boxes,dtype=np.float)
    print('boxes:',boxes)

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    #Do not use RPN.
    #cfg.MODEL.FASTER_RCNN = False

    dataset = datasets.get_coco_dataset()
    cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    assert bool(args.load_ckpt) ^ bool(args.load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN_predictor_with_boxes = Generalized_RCNN_Predictor_with_Boxes()

    if args.cuda:
        maskRCNN_predictor_with_boxes.cuda()

    if args.load_ckpt:
        load_name = args.load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN_predictor_with_boxes, checkpoint['model'])

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN_predictor_with_boxes, args.load_detectron)

    maskRCNN_predictor_with_boxes = mynn.DataParallel(maskRCNN_predictor_with_boxes, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN_predictor_with_boxes.eval()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert im is not None

    timers = defaultdict(Timer)

    cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN_predictor_with_boxes, im, timers=timers,box_proposals = boxes)
    im_name = 'temp_name'
    
    vis_utils.vis_one_image(
        im[:, :, ::-1],  # BGR -> RGB for visualization
        im_name,
        args.output_dir,
        cls_boxes,
        cls_segms,
        cls_keyps,
        dataset=dataset,
        box_alpha=0.3,
        show_class=True,
        thresh=0.7,
        kp_thresh=2
    )

if __name__ == '__main__':
    main()
