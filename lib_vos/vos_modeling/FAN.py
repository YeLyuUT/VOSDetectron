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
from vos_nn.convgrucell import ConvGRUCell2d
from vos_model import flow_align

class FAN_GRU(nn.Module):
  """

  """
  def __init__(self,i_channels,o_channels,scale):
    super().__init__()
    d = cfg.CONVGRU
    
    self.i_channels = i_channels
    self.h_channels = d.HIDDEN_STATE_CHANNELS
    self.kernel_size = d.KERNEL_SIZE
    self.stride=d.STRIDE
    self.dilation=d.DILATION
    self.groups=d.GROUPS
    self.use_GN = d.USE_GN
    self.GN_groups = d.GN_GROUPS
    self.conv_gru_cell = ConvGRUCell2d(i_channels,self.h_channels,kernel_size=self.kernel_size, stride=self.stride, 
      dilation=self.dilation, groups=self.groups, use_GN=self.use_GN,GN_groups = self.GN_groups)

    


