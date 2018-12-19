import os.path as osp
import sys
import os
import numpy as np
import json
import glob

sys.path.append(osp.abspath('../..'))
from imdb.imdb import imdb as imdb
DEBUG = False

class cityscape_semantic_db(imdb):
  def __init__(self,cityscape_root):
    super(cityscape_semantic_db,self).__init__('cityscape_semantic')
    self.root = cityscape_root
    
  def getFineImageLabelPair(self,split):
    splits=['train','test','val']
    assert(split in splits)
    label_path_template = osp.join(self.root,'gtFine',split,'*','*labelTrainIds.png')
    label_paths_candidates = glob.glob(label_path_template)
    if len(label_paths_candidates)==0:
      raise Exception('Cannot find fine labels. Have downloaded them?')
    image_paths = []
    label_paths = []
    for lbl_p in label_paths_candidates:
      idx = lbl_p.find(self.root)
      img_p = lbl_p[:idx]+lbl_p[idx:].replace('gtFine_labelTrainIds','leftImg8bit').replace('gtFine','leftImg8bit')
      if osp.exists(img_p) and img_p!=lbl_p:
        image_paths.append(img_p)
        label_paths.append(lbl_p)
      else:
        raise Exception('image path:%s does not exist for label:%s'%(img_p,lbl_p))

    assert(len(image_paths)==len(label_paths))
    print('Finish preparation for image label pairs.')
    print('Total number of image label pairs is %d'%(len(image_paths)))

    return image_paths,label_paths 
  
  def getCoarseImageLabelPair(self,split):
    splits=['train','train_extra','val']
    assert(split in splits)
    label_path_template = osp.join(self.root,'gtCoarse',split,'*','*labelTrainIds.png')
    label_paths_candidates = glob.glob(label_path_template)
    if len(label_paths_candidates)==0:
      raise Exception('Cannot find coarse labels. Have downloaded them?')
    image_paths = []
    label_paths = []
    for lbl_p in label_paths_candidates:
      idx = lbl_p.find(self.root)
      img_p = lbl_p[:idx]+lbl_p[idx:].replace('gtCoarse_labelTrainIds','leftImg8bit').replace('gtCoarse','leftImg8bit')
      if osp.exists(img_p) and img_p!=lbl_p:
        image_paths.append(img_p)
        label_paths.append(lbl_p)
      else:
        raise Exception('image path:%s does not exist for label:%s'%(img_p,lbl_p))

    assert(len(image_paths)==len(label_paths))
    print('Finish preparation for image label pairs.')
    print('Total number of image label pairs is %d'%(len(image_paths)))

    return image_paths,label_paths 
      
  def getTrainData(self,with_extra = False):
    img_list = []
    lbl_list = []
    imgs,lbls = self.getFineImageLabelPair('train')
    img_list.extend(imgs)
    lbl_list.extend(lbls)
    if with_extra:
      imgs,lbls = self.getCoarseImageLabelPair('train_extra')
      img_list.extend(imgs)
      lbl_list.extend(lbls)

    if DEBUG:
      for i in range(10):
        print(img_list[i])
        print(lbl_list[i])
      if with_extra:
        for i in range(3010,3020):
          print(img_list[i])
          print(lbl_list[i])
    return img_list,lbl_list

  def getValData(self):
    img_list = []
    lbl_list = []
    imgs,lbls = self.getFineImageLabelPair('val')
    img_list.extend(imgs)
    lbl_list.extend(lbls)
    if DEBUG:
      for i in range(10):
        print(img_list[i])
        print(lbl_list[i])

    return img_list,lbl_list


def test(root):
  db = cityscape_semantic_db(root)
  db.getTrainData(True)

if __name__=='__main__':
  root = '/media/yelyu/Seagate Expansion Drive/researchData/dataset/Cityscape'
  test(root)
    
    
