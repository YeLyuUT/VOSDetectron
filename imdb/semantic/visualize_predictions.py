import sys
cityscape_home = '/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Dataset/Cityscape/Cityscape/cityscapesscripts/'
import os
from os import path as osp
import numpy as np

if not cityscape_home in sys.path:
  sys.path.append(cityscape_home)

from helpers import labels as lbl
from PIL import Image
import time

def trainIdToColorDict():
  ref_dict = {}
  for l in lbl.labels:
    ref_dict[l.trainId]=l.color
  return ref_dict

ref_dict = trainIdToColorDict()

def getBlendImageOutputPath(pred_img_path,output_dir = None):
  if output_dir is not None:
    return osp.join(output_dir,osp.basename(pred_img_path.replace('*','_blend')))
  else:
    return pred_img_path.replace('*','_blend')

def getColorImageOutputPath(pred_img_path,output_dir = None):
  if output_dir is not None:
    return osp.join(output_dir,osp.basename(pred_img_path.replace('*','_clr')))
  else:
    return pred_img_path.replace('*','_clr')

def trainIdImgToColorImg(trainIdImgPath):
  tic = time.time()
  print('tic',tic)
  img = np.array(Image.open(trainIdImgPath))
  img_out = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
  vals = np.unique(img.reshape([-1]))
  for v in vals:
    mask = v==img
    img_out[mask] = ref_dict[v]
  toc = time.time()
  print('toc',toc)
  print('Time elapse:%f'%(toc-tic))
  return img_out

if __name__=='__main__':
  print(sys.argv)
  txtfile = sys.argv[1]
  viz_folder = sys.argv[2]
  _items_list = []
  with open(txtfile,'r') as f:
      lines = f.readlines()
      for line in lines:
        items = line.split()
        assert(len(items)==3)          
        assert(osp.exists(items[0]))
        _items_list.append(items)
  for item in _items_list:
    clr_lbl = trainIdImgToColorImg(item[2])    
    clr_path = getColorImageOutputPath(item[2])
    blend_path = getBlendImageOutputPath(item[2])
    assert(clr_path!=item[2])
    if not osp.isdir(viz_folder):
      print('create folder:%s'%(viz_folder))
      os.mkdirs(viz_folder)
    outpath = getColorImageOutputPath(item[2],viz_folder)
    Image.fromarray(clr_lbl).save(outpath)
    outpath = getBlendImageOutputPath(item[2],viz_folder)
    Image.fromarray(np.array(np.array(Image.open(item[0]))*0.5+clr_lbl*0.5,dtype=np.uint8)).save(outpath)
