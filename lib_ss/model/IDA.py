import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import nn as mynn
from .config import cfg as SS_config

DEBUG = True

def IDA_ResNet50_conv5_body():
  return IDA()

class IDA_level(nn.Module):
  """ IDA structure in one level.
  
  
  """
  def __init__(self,in_dims,out_dims):
    super(IDA_level,self).__init__()
    assert(len(in_dims)==len(out_dims)+1)
    self.upsampleModules = nn.ModuleList()
    self.lateralModules = nn.ModuleList()
    self.relu = nn.ReLU(inplace=True)
    for idx in range(len(in_dims)-1):      
      #upsample to 2X size
      i_dim = in_dims[idx+1]
      o_dim = out_dims[idx]
      self.upsampleModules.append(
      nn.Sequential(
      nn.Conv2d(i_dim,o_dim,kernel_size=1,bias=False),
      nn.GroupNorm(num_groups=32,num_channels=o_dim),
      nn.UpsamplingBilinear2d(scale_factor=2))
      )
      #lateral convolution
      i_dim = in_dims[idx]
      o_dim = out_dims[idx]
      self.lateralModules.append(
      nn.Sequential(
      nn.Conv2d(i_dim,o_dim,kernel_size=1,bias=False),
      nn.GroupNorm(num_groups=32,num_channels=o_dim))
      )

  def _init_weights(self):
    def init_func(m):
      if DEBUG:
        print(type(m))
      if isinstance(m, nn.Conv2d):
        mynn.init.XavierFill(m.weight)
        if DEBUG:
          print('XavierFill')
        if m.bias is not None:
          init.constant_(m.bias, 0)
      else:
        if DEBUG:
          print('No init')
    for child_m in self.upsampleModules.children():
      child_m.apply(init_func)
      
  def forward(self,*Xs):
    assert(len(Xs)==len(in_dims))
    
    

class IDA(nn.Module):
  """ IDA module for semantic segmentation.
  
  
  """
  def __init__(self,conv_body_func,IDA_gating = False):
    super(IDA,self).__init__()
    self.IDA_gating = IDA_gating
    self._init_weights()
    self.conv_body = conv_body_func()
    self.conv_level1 = IDA_level(SS_config.LEVEL0.out_dims,SS_config.LEVEL1.out_dims)
    self.conv_level2 = IDA_level(SS_config.LEVEL1.out_dims,SS_config.LEVEL2.out_dims)
    self.conv_level3 = IDA_level(SS_config.LEVEL2.out_dims,SS_config.LEVEL3.out_dims)
    
  def _init_weights(self):
    def init_func(m):
      if isinstance(m,nn.Conv2d):
        mynn.init.XavierFill(m.weight)
        if m.bias is not None:
            init.constant_(m.bias, 0)
            
    for child_m in self.children():
      #ModuleList contains modules that are initialized by themselves
      if not isinstance(child_m,nn.ModuleList):
        child_m.apply(init_func)
        
