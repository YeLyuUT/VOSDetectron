import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable
from core.config import cfg
import nn as mynn

class SelfAttention(nn.Module):
  def __init__(self,dim):
    super().__init__()
    self.B = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, padding=1, dilation=1, bias=False), nn.ReLU())
    self.C = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, padding=1, dilation=1, bias=False), nn.ReLU())
    self.D = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, padding=1, dilation=1, bias=False), nn.ReLU())
    self._init_weights()

  def _init_weights(self):
    def init_func(m):
      if isinstance(m, nn.Conv2d):
          if cfg.MRCNN.CONV_INIT == 'GaussianFill':
              init.normal_(m.weight, std=0.001)
          elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
              mynn.init.MSRAFill(m.weight)
          else:
              raise ValueError
          if m.bias is not None:
              init.constant_(m.bias, 0)
    for m in self.children():
      m.apply(init_func)

  def forward(self, x):
    feat1 = self.B(x)
    feat2 = self.C(x)
    feat3 = self.D(x)
    out = feat3.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).matmul(
      feat1.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).transpose(1,2).matmul(feat2.view(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
      ).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    return out+x

class SimpleAttention(nn.Module):
  def __init__(self,dim):
    super().__init__()
    self.B = nn.Sequential(nn.Conv2d(dim, dim, 3, 1, padding=2, dilation=2, bias=True), nn.ReLU(),
      nn.Conv2d(dim, dim, 3, 1, padding=2, dilation=2, bias=True), nn.ReLU(),
      nn.Conv2d(dim, dim, 3, 1, padding=2, dilation=2, bias=True), nn.ReLU())
    self._init_weights()

  def _init_weights(self):
    def init_func(m):
      if isinstance(m, nn.Conv2d):
          if cfg.MRCNN.CONV_INIT == 'GaussianFill':
              init.normal_(m.weight, std=0.001)
          elif cfg.MRCNN.CONV_INIT == 'MSRAFill':
              mynn.init.MSRAFill(m.weight)
          else:
              raise ValueError
          if m.bias is not None:
              init.constant_(m.bias, 0)
      elif isinstance(m, nn.Sequential):
        for mm in m.children():
          mm.apply(init_func)
    for m in self.children():
      m.apply(init_func)

  def forward(self, x):
    return x+x*self.B(x)