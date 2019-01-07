import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from PIL import Image
import numpy as np
from os import path as osp
from enum import Enum
from .ss_Transformer import Transformer as Transformer
from .ss_Transformer import Transformer_equally_spaced_crop as Transformer_equally_spaced_crop
import PIL

import sys
from os import path as osp
import numpy as np

dir_path = osp.dirname(osp.realpath(__file__))
sys.path.append(osp.abspath(osp.join(dir_path,'../ss_core')))
from ss_config import cfg as config

class Mode(Enum):
  TRAIN=1
  VAL_NO_PRED=2
  VAL_PRED = 3
  TEST=4

class SemanticSegmentationDataset(data.Dataset):
  def __init__(self, txtfile, mode, use_data_augmentation = True, assertNum = None, gtMapper = None, batch_size = 1):
    """
    Args:
    txtfile: txt file path of which contains image label paths.
              For training mode or validation mode without prediction:
                  txt file should contain tuples like rows, each row is  "<imgPath> <groundTruthPath>\n"
              For validation mode with prediction:
                  txt file should contain tuples like rows, each row is  "<imgPath> <groundTruthPath> <outputPath>\n"
              For test mode:
                  txt file should contain tuples like rows, each row is  "<imgPath> <outputPath>\n"
    assertNum: if not None, shuold be a number to check if correct number of items are read.
    gtMapper: mapper that changes gt values
    """
    assert(isinstance(mode,Mode))
    self._items_list =  []
    self._mode = mode
    self._use_data_augmentation = use_data_augmentation
    self._gtMapper = gtMapper
    self._batch_size = batch_size
    with open(txtfile,'r') as f:
      lines = f.readlines()
      for line in lines:
        items = line.split()
        if len(items)!=2 and len(items)!=3:
          #invalide line.
          continue
        if self._mode==Mode.TRAIN or self._mode==Mode.VAL_NO_PRED:
          assert(len(items)==2)
          assert(osp.exists(items[0]))
          assert(osp.exists(items[1]))
        elif self._mode==Mode.VAL_PRED:
          assert(len(items)==3)
          assert(osp.exists(items[0]))
          assert(osp.exists(items[1]))
        elif self._mode==Mode.TEST:
          assert(len(items)==3)          
          assert(osp.exists(items[0]))
        self._items_list.append(items)
    if assertNum is not None:
      assert(assertNum==len(self._items_list))
  
  def get_output_path(self,index):
    if self._mode==Mode.VAL_PRED:
      return self._items_list[index][2]
    if self._mode==Mode.TEST:
      return self._items_list[index][1]
  
  def get_Transformer(self):
    return self._TF
  
  def __getitem__(self, index):
    items = self._items_list[index]
    inputs = {'img_PIL': None, 'gt_PIL': None, 'index': None, 'pred_path':None}
    for item in items:
      print(osp.basename(item))
    if self._mode==Mode.TRAIN or self._mode==Mode.VAL_NO_PRED:
      imgPath = items[0]
      gtPath = items[1]
      img_PIL = Image.open(imgPath)
      gt_PIL = Image.open(gtPath)
      inputs['img_PIL'] = img_PIL
      inputs['gt_PIL'] = gt_PIL
      inputs['index'] = np.array(index,np.int32)
    elif self._mode==Mode.VAL_PRED:
      imgPath = items[0]
      gtPath = items[1]
      img_PIL = Image.open(imgPath)
      gt_PIL = Image.open(gtPath)
      inputs['img_PIL'] = img_PIL
      inputs['gt_PIL'] = gt_PIL
      inputs['index'] = np.array(index,np.int32)
      inputs['pred_path'] = items[2]
    elif self._mode==Mode.TEST:
      imgPath = items[0]
      img_PIL = Image.open(imgPath)
      inputs['img_PIL'] = img_PIL
      inputs['index'] = np.array(index,np.int32)
      inputs['pred_path'] = items[2]
    else:
      raise Exception('What??? This is impossible.')
    if self._mode==Mode.TRAIN:
      #Data augmentation.
      TF = Transformer(config.TRAIN.INPUT_BLOB_SIZE)
      if not self._use_data_augmentation:
        TF.reset_TF_dict()
      inputs['img_PIL'],inputs['gt_PIL'] = TF(inputs['img_PIL'],inputs['gt_PIL'],batch_size = self._batch_size)
    elif self._mode==Mode.VAL_NO_PRED or self._mode==Mode.VAL_PRED or self._mode==Mode.TEST:
      TF = Transformer_equally_spaced_crop(config.TRAIN.INPUT_BLOB_SIZE)
      inputs['img_PIL'],inputs['gt_PIL'] = TF(inputs['img_PIL'],inputs['gt_PIL'])
    self._TF = TF
    
    #generate Tensor
    inputs['index'] = torch.from_numpy(inputs['index'])
    if inputs['img_PIL'] is not None:
      inputs['img_PIL'] = np.array(inputs['img_PIL'],dtype=np.float32)
      #normalize image to range 0-1
      inputs['img_PIL'] = inputs['img_PIL']*2./255.-1.
      inputs['img_PIL'] = np.transpose(inputs['img_PIL'], axes=(0,3,1,2))
      #print(inputs['img_PIL'].shape)
      inputs['img_PIL'] = torch.from_numpy(inputs['img_PIL'])
    if inputs['gt_PIL'] is not None:
      inputs['gt_PIL'] = np.array(inputs['gt_PIL'],dtype=np.int64)
      if not self._gtMapper is None:
        inputs['gt_PIL'] = self._gtMapper(inputs['gt_PIL'])
      inputs['gt_PIL'] = torch.from_numpy(inputs['gt_PIL'])
    for name in inputs.keys():
      #handle None type for pytorch.      
      if inputs[name] is None:
        inputs[name] = []
    return inputs

  def __len__(self):
    return len(self._items_list)
