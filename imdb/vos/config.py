from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
from packaging import version
import torch
import torch.nn as nn
from torch.nn import init
import yaml

from easydict import EasyDict as AttrDict

__C = AttrDict()
cfg = __C
__C.DEBUG = False
__C.CACHE_DIR = osp.abspath(osp.join(osp.dirname(__file__),'./cache'))
__C.COCO_API_HOME = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/MSCOCO/PythonAPI'
# ---------------------------------------------------------------------------- #
# DAVIS
# ---------------------------------------------------------------------------- #
#directory contains davis api tools.
__C.DAVIS = AttrDict()
__C.DAVIS.HOME = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/DAVIS/davis-2017'
__C.DAVIS.FLOW_DIR = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Opensrc_proj/LiteFlowNet/models/testing/davis_flow_backward'
__C.DAVIS.FLOW_INV_DIR = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Opensrc_proj/LiteFlowNet/models/testing/davis_flow_forward'
__C.DAVIS.FLOW_FILENAME_TEMPLATE = 'liteflownet-%07d.flo'

# ---------------------------------------------------------------------------- #
# SegTrack v2
# ---------------------------------------------------------------------------- #
__C.SegTrack_v2 = AttrDict()
__C.SegTrack_v2.HOME = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/SegTrackv2'
__C.SegTrack_v2.FLOW_DIR = None
__C.SegTrack_v2.FLOW_INV_DIR = None
__C.SegTrack_v2.FLOW_FILENAME_TEMPLATE = 'liteflownet-%07d.flo'