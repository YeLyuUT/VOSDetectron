import torch
import torch.nn as nn
import numpy as np
import sys
import os
from os import path as osp
from PIL import Image

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

def test_down_sample():
  vos_util_path = osp.abspath('../../../vos_utils')
  if vos_util_path not in sys.path:
    sys.path.append(vos_util_path)

  from flow_util.readFlowFile import read as readflo
  from flow_util.writeFlowFile import write as writeflo
  spatial_scale = 0.5
  conv_flow_downsample = _flow_downsample_convolutional_layer(spatial_scale)
  dir_name = 'car-race'
  inPath = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Work/ICCV19/flow/DAVIS/Flow/480p/'+dir_name+'/00000_forward.flo'
  outPath = './'+dir_name+'.flo'
  flow_in = readflo(inPath)
  flow_in = flow_in.transpose([2,0,1])
  flow_in = np.expand_dims(flow_in,axis=0)
  print(flow_in.shape)
  tensor_in = torch.tensor(flow_in, requires_grad=True).cuda()
  print(tensor_in.shape)
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


def test_flow_align():
  vos_util_path = osp.abspath('../../../vos_utils')
  if vos_util_path not in sys.path:
    sys.path.append(vos_util_path)
  from flow_util.readFlowFile import read as readflo
  from flow_util.writeFlowFile import write as writeflo

  vos_model_path = osp.abspath('../../..')
  print(vos_model_path)
  if vos_model_path not in sys.path:
    sys.path.append(vos_model_path)
  from vos_model.flow_align.modules.flow_align import FlowAlign
  
  dir_name = 'blackswan'
  inFlowPath = osp.join('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Opensrc_proj/LiteFlowNet/models/testing/davis_flow_backward',dir_name,'liteflownet-0000000.flo')
  inImgPath = osp.join('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/DAVIS/davis-2017/data/DAVIS/JPEGImages/480p',dir_name,'00000.jpg')
  inImgPath2 = osp.join('/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/DAVIS/davis-2017/data/DAVIS/JPEGImages/480p',dir_name,'00001.jpg')
  outImgPath = './out1.png';
  outImgPath2 = './out3.png';
  outPath = './out2.png';
  flow_in = readflo(inFlowPath)
  g_h = 202
  g_w = 202
  sz = 40
  flow_in = flow_in.astype(np.float32)[g_h:g_h+sz,g_w:g_w+sz,:]
  flow_in = flow_in.transpose([2,0,1])
  flow_in = np.expand_dims(flow_in,axis=0)
  tensor_in_flo = torch.tensor(flow_in, requires_grad=True).cuda()
  npImg = np.array(Image.open(inImgPath))
  npImg = npImg[g_h:g_h+sz,g_w:g_w+sz,:]
  npImg = np.array(Image.fromarray(npImg).resize((npImg.shape[1]//2,npImg.shape[0]//2), Image.ANTIALIAS),dtype=np.float32)
  npImg = npImg.transpose([2,0,1])
  npImg = np.expand_dims(npImg,axis=0)
  tensor_in_im = torch.tensor(npImg, requires_grad=True).cuda()
  FAL = FlowAlign(0.5).cuda()
  print(tensor_in_im.shape)
  print(tensor_in_flo.shape)
  print('grad_check:',grad_check(FAL,(tensor_in_im,tensor_in_flo)))
  #print('grad_check:',my_grad_check(FAL,(tensor_in_im,tensor_in_flo)))
  
  '''
  tensor_out_im = FAL(tensor_in_im,tensor_in_flo)
  im_out = tensor_out_im.data.cpu().numpy()
  im_out = np.squeeze(im_out,axis=0)
  im_out = im_out.transpose([1,2,0])
  print(im_out.shape)
  Image.fromarray(im_out.astype(np.uint8)).save(outPath)
  Image.open(inImgPath).save(outImgPath)
  Image.open(inImgPath2).save(outImgPath2)
  '''

def grad_check(func,inputs):
  from torch.autograd import gradcheck
  return gradcheck(func,inputs,eps=1e-2,atol=1e-2)

if __name__=='__main__':
  #test_down_sample()
  test_flow_align()
  
  
  
  
  
  
  
