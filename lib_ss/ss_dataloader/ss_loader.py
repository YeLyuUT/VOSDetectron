import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from PIL import Image
import numpy as np
from os import path as osp
from enum import Enum
import torchvision.transforms.functional as TF
import PIL

class Mode(Enum):
  TRAIN=1
  VAL_NO_PRED=2
  VAL_PRED = 3
  TEST=4

class SemanticSegmentationDataset(data.Dataset):
  def __init__(self, txtfile, mode, assertNum = None):
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
    """
    assert(isintance(mode,Mode))
    self._items_list =  []
    self._mode = mode
    with open(txtfile,'r') as f:
      lines = f.readlines()
      for line in lines:
        items = line.split(' ')
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
          assert(len(items)==2)
          assert(osp.exists(items[0]))
        self._items_list.append(items)
    if assertNum is not None:
      assert(assertNum==len(self._items_list))

  def __getitem__(self, index):
    items = self._items_list[index]
    blobs = {}
    if self._mode==Mode.TRAIN or self._mode==Mode.VAL_NO_PRED:
      imgPath = items[0]
      gtPath = items[1]
      img_blob = torch.from_numpy(np.array(Image.open(imgPath)).astype(np.float32))
      gt_blob = torch.from_numpy(np.array(Image.open(gtPath)).astype(np.int32))
      blobs['img_blob'] = img_blob
      blobs['gt_blob'] = gt_blob
    elif self._mode==Mode.VAL_PRED:
      imgPath = items[0]
      gtPath = items[1]
      img_blob = torch.from_numpy(np.array(Image.open(imgPath)).astype(np.float32))
      gt_blob = torch.from_numpy(np.array(Image.open(gtPath)).astype(np.int32))
      blobs['img_blob'] = img_blob
      blobs['gt_blob'] = gt_blob
      blobs['output_path'] = items[2]
    elif self._mode==Mode.TEST:
      imgPath = items[0]
      img_blob = torch.from_numpy(np.array(Image.open(imgPath)).astype(np.float32))
      blobs['img_blob'] = img_blob
      blobs['output_path'] = items[2]
    else:
      raise Exception('What??? This is impossible.')

  def __len__(self):
    return len(self._items_list)
