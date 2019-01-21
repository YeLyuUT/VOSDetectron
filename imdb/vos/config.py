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
__C.DEBUG = True
__C.CACHE_DIR = './cache'
__C.COCO_API_HOME = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/MSCOCO/PythonAPI'
# ---------------------------------------------------------------------------- #
# DAVIS
# ---------------------------------------------------------------------------- #
#directory contains davis api tools.
__C.DAVIS = AttrDict()
__C.DAVIS.HOME = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/DAVIS/davis-2017'


