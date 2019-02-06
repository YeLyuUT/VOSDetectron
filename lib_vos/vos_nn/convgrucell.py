import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Variable

import nn as mynn

class ConvGRUCell2d(nn.Module):
  """
  This implementation keeps spatial size of the input blob and output blob.
  Args:
  i_channels: input blob channels.
  h_channels: output blob channels of inner gates.
  kernel_size: same as 2D convolution.
  stride: same as 2D convolution.
  dilation: same as 2D convolution.
  groups: same as 2D convolution.
  use_GN: if True, use group norm. else use bias.
  """
  def __init__(self,i_channels,h_channels,kernel_size=3, stride=1, dilation=1, groups=1, use_GN=True,GN_groups = 32):
    super().__init__()
    self.use_GN = use_GN
    self.i_channels = i_channels
    self.h_channels = h_channels
    self.kernel_size = kernel_size
    self.stride=stride
    self.dilation=dilation
    self.groups=groups
    self.GN_groups = GN_groups

    padding=kernel_size//2
    #update gate
    self.Wz_h = nn.Conv2d(h_channels,h_channels,kernel_size=1,padding=0, stride=stride, dilation=dilation, groups=groups, bias=False)
    self.Wz_x = nn.Conv2d(i_channels,h_channels,kernel_size=kernel_size,padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
    #reset gate  
    self.Wr_h = nn.Conv2d(h_channels,h_channels,kernel_size=1,padding=0, stride=stride, dilation=dilation, groups=groups, bias=False)
    self.Wr_x = nn.Conv2d(i_channels,h_channels,kernel_size=kernel_size,padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
    #hidden state
    self.Wh_h  = nn.Conv2d(h_channels,h_channels,kernel_size=1,padding=0, stride=stride, dilation=dilation, groups=groups, bias=False)
    self.Wh_x  = nn.Conv2d(i_channels,h_channels,kernel_size=kernel_size,padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
    #non-linear ops
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    #bias
    if not self.use_GN:
      self.bz = nn.Parameter(torch.Tensor(h_channels))
      self.br = nn.Parameter(torch.Tensor(h_channels))
      self.bh = nn.Parameter(torch.Tensor(h_channels))
    else:
      self.bz = nn.GroupNorm(GN_groups,h_channels)
      self.br = nn.GroupNorm(GN_groups,h_channels)
      self.bh = nn.GroupNorm(GN_groups,h_channels)

    self._init_weights()

  def _init_weights(self):
    #init bias
    if not self.use_GN:
      init.constant_(self.bz,0)
      init.constant_(self.br,0)
      init.constant_(self.bh,0)

    def init_func(m):
      if isinstance(m,nn.Conv2d):
        mynn.init.XavierFill(m.weight)
        if m.bias is not None:
          init.constant_(m.bias,0)
    for m in self.children():
      m.apply(init_func)

  def forward(self,x):
    """
    Args:
    x: input tuple x[0] is input, x[1] is hidden state
    """
    _input = x[0]
    _hidden_state = x[1]
    z = None
    h_ = None
    if not self.use_GN:
      z = self.sigmoid(self.Wz_h(_hidden_state)+self.Wz_x(_input)+self.bz)
      r = self.sigmoid(self.Wr_h(_hidden_state)+self.Wr_x(_input)+self.br)
      h_ = self.tanh(self.Wh_h(torch.mul(_hidden_state,r))+self.Wh_x(_input)+self.bh)
    else:
      z = self.sigmoid(self.bz(self.Wz_h(_hidden_state)+self.Wz_x(_input)))
      r = self.sigmoid(self.br(self.Wr_h(_hidden_state)+self.Wr_x(_input)))
      h_ = self.tanh(self.bh(self.Wh_h(torch.mul(_hidden_state,r))+self.Wh_x(_input)))

    h = torch.mul(1-z,_hidden_state)+torch.mul(z,h_)
    return h

  def extra_repr(self):
    # (Optional)Set the extra information about this module. You can test
    # it by printing an object of this class.
    self.use_GN = use_GN
    self.i_channels = i_channels
    self.h_channels = h_channels
    self.kernel_size = kernel_size
    self.GN_groups = GN_groups
    return 'i_channels={}, h_channels={}, kernel_size={}, use_GN={}, GN_groups={}'.format(
        self.i_channels, self.h_channels, self.kernel_size, self.use_GN, self.GN_groups)
