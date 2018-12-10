"""
Flow alignment network

Two types:
1) ConvGRU version
2) ConvLSTM version

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

from core.config import cfg

class FAN_GRU(nn.Module):
