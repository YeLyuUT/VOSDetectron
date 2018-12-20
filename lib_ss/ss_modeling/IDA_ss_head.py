import torch
import torch.nn as nn
import os
import sys
import os.path as osp
dir_path = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.abspath(osp.join(dir_path,'../../llib')))
import initialization as init
from ss_config import cfg as config

class IDA_ss_outputs(nn.Module):
  def __init__(self, class_num):
    super(IDA_ss_outputs,self).__init__()
    self.class_num = class_num
    self.up_sample_list = nn.ModuleList()
    self.scales = config.IDA.SCALES
    for scale in self.scales:
      self.up_sample_list.append(self._upsample_layer(scale))
    self.pred = self._prediction_layer()
    
  def _upsample_layer(self, scale_factor):
    return nn.UpsamplingBilinear2d(scale_factor=scale_factor)

  def _prediction_layer(self):
    i_dim = config.IDA.LEVEL0_out_dims[-1]\
           +config.IDA.LEVEL1_out_dims[-1]\
           +config.IDA.LEVEL2_out_dims[-1]\
           +config.IDA.LEVEL3_out_dims[-1]
    o_dim = self.class_num
    return nn.Conv2d(i_dim,o_dim,kernel_size=1,bias=False)

  def forward(self, blobs):
    assert(blobs is list)
    assert(len(blobs)==len(self.scales))
    same_scale_blobs = []
    for idx in range(len(blobs)):
      same_scale_blobs.append(self.up_sample_list(blobs[idx]))
    cat_blob = torch.cat(same_scale_blobs,dim=1)
    top_blob = self.pred(cat_blob)
    return top_blob