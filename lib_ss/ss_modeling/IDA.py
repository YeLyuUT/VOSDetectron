import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import nn as mynn
import os.path as osp
import os
import sys
dir_path = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.abspath(osp.join(dir_path,'../ss_core')))
sys.path.append(osp.abspath(osp.join(dir_path,'../../lib')))
sys.path.append(osp.abspath(osp.join(dir_path,'../../llib')))

from initialization import init_func as init_func
from ss_config import cfg as config
import modeling.ResNet as ResNet

DEBUG = True

#def IDA_ResNet50_conv5_body():
# return IDA()

def IDA_ResNet50_conv5_body():
  return IDA(ResNet.ResNet50_conv5_body)

def IDA_ResNet101_conv5_body():
  return IDA(ResNet.ResNet101_conv5_body)

class IDA_level(nn.Module):
  """ IDA structure in one level.
  
  
  """
  def __init__(self,in_dims,out_dims):
    super(IDA_level,self).__init__()
    assert(len(in_dims)==len(out_dims)+1)
    self.upsampleModules = nn.ModuleList()
    self.lateralModules = nn.ModuleList()
    self.relu = nn.ReLU(inplace=True)
    self.in_dims = in_dims
    for idx in range(len(in_dims)-1):      
      #upsample to 2X size
      i_dim = in_dims[idx+1]
      o_dim = out_dims[idx]
      self.upsampleModules.append(
      nn.Sequential(
      nn.Conv2d(i_dim,o_dim,kernel_size=1,bias=False),
      nn.GroupNorm(num_groups=2,num_channels=o_dim),
      nn.UpsamplingBilinear2d(scale_factor=2))
      )
      #lateral convolution
      i_dim = in_dims[idx]
      o_dim = out_dims[idx]
      self.lateralModules.append(
      nn.Sequential(
      nn.Conv2d(i_dim,o_dim,kernel_size=1,bias=False),
      nn.GroupNorm(num_groups=2,num_channels=o_dim))
      )
    self._init_weights()

  def _init_weights(self):
    for child_m in self.upsampleModules.children():
      child_m.apply(init_func)
    for child_m in self.lateralModules.children():
      child_m.apply(init_func)
      
  def forward(self,Xs):
    assert(len(Xs)==len(self.in_dims))
    top_blobs = []
    for idx in range(len(self.in_dims)-1):
      X1 = Xs[idx]
      X2 = Xs[idx+1]
      top_blobs.append(self.relu(self.upsampleModules[idx](X2)+self.lateralModules[idx](X1)))
    return top_blobs

class IDA(nn.Module):
  """ IDA module for semantic segmentation.
  
  
  """
  def __init__(self,conv_body_func,IDA_gating = False):
    super(IDA,self).__init__()
    self.IDA_gating = IDA_gating
    self.conv_body = conv_body_func()
    self.IDA_level1 = IDA_level(config.IDA.LEVEL0_out_dims,config.IDA.LEVEL1_out_dims)
    self.IDA_level2 = IDA_level(config.IDA.LEVEL1_out_dims,config.IDA.LEVEL2_out_dims)
    self.IDA_level3 = IDA_level(config.IDA.LEVEL2_out_dims,config.IDA.LEVEL3_out_dims)
    
    
  def forward(self, x):
    conv_body_blobs = [self.conv_body.res1(x)]
    for i in range(1, self.conv_body.convX):
      conv_body_blobs.append(getattr(self.conv_body, 'res%d' % (i+1))(conv_body_blobs[-1]))
    IDA_blobs = []
    IDA_blobs.append(self.IDA_level1(conv_body_blobs[1:]))
    IDA_blobs.append(self.IDA_level2(IDA_blobs[-1]))
    IDA_blobs.append(self.IDA_level3(IDA_blobs[-1]))

    IDA_top_blobs = [conv_body_blobs[-1]]
    for i in range(len(IDA_blobs)):
      IDA_top_blobs.append(IDA_blobs[i][-1])
    return IDA_top_blobs

  
