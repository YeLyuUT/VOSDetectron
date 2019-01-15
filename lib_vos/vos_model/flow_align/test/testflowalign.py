import torch
import torch.nn as nn
import numpy as np
import sys
import os
from os import path as osp
vos_util_path = osp.abspath('../../../vos_utils')
if vos_util_path not in sys.path:
  sys.path.append(vos_util_path)

from flow_util.readFlowFile import read as readflo
from flow_util.writeFlowFile import write as writeflo

def _flow_downsample_convolutional_layer(spatial_scale):
  assert(spatial_scale<=1)
  inv_scale = int(1.0/spatial_scale)
  kernel_size = inv_scale
  stride = inv_scale
  conv_flow_downsample = nn.Conv2d(2,2,kernel_size = kernel_size,stride = stride,padding = 0, dilation = 1,groups = 1,bias = False)
  init_weights = torch.zeros(conv_flow_downsample.weight.shape)
  for idx in range(init_weights.shape[0]):
    init_weights[idx,idx,:,:] = spatial_scale**3
  conv_flow_downsample.weight = torch.nn.Parameter(init_weights)
  return conv_flow_downsample


if __name__=='__main__':
  spatial_scale = 0.5
  conv_flow_downsample = _flow_downsample_convolutional_layer(spatial_scale)
  dir_name = 'bike-packing'
  inPath = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Work/ICCV19/flow/DAVIS/Flow/480p/'+dir_name+'/00000_forward.flo'
  outPath = './'+dir_name+'.flo'
  flow_in = readflo(inPath)
  flow_in = flow_in.transpose([2,0,1])
  flow_in = np.expand_dims(flow_in,axis=0)
  print(flow_in.shape)
  tensor_in = torch.tensor(flow_in).cuda()
  conv_flow_downsample = conv_flow_downsample.cuda()
  tensor_out = conv_flow_downsample(tensor_in)
  torch.cuda.empty_cache()
  flow_out = tensor_out.data.cpu().numpy()
  print(flow_out.shape)
  flow_out = np.squeeze(flow_out,axis=0)
  flow_out = flow_out.transpose([1,2,0])
  print(flow_out.shape)
  print(flow_out.dtype)
  writeflo(flow_out,outPath)
  
  
  
  
  
  
  
