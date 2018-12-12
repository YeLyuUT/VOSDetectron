from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
from ps_cnn.config import cfg
import os.path as osp
import sys
import os
import numpy as np
import scipy.sparse
import scipy.io as sio
import cPickle
import json
import uuid

DEBUG = False

class coco_ps_db(imdb):
  def __init__(self,split,year):
    imdb.__init__(self,'coco_panoptic_'+split+year)
    assert(split in ['train','val','test'])
    self._split = split
    self._year = year

    self._coco_img_dir = self._get_coco_image_dir(split,year)
    self._coco_seg_dir = self._get_coco_seg_dir(split,year)
    self._coco_anns_json = self._get_coco_anns_json(split,year)

    with open(coco_anns_json, 'r') as f:
      self._coco_d = json.load(f)

    self._classes = tuple(['__background__']+[c['name'] for c in self._coco_d['categories']])
    self._coco_cat_ids = tuple([0]+[c['id'] for c in self._coco_d['categories']])

    self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
    self._ind_to_class = dict(zip(xrange(self.num_classes), self.classes))

    self._class_to_coco_cat_id = dict(zip([c['name'] for c in cats],self._coco_cat_ids))
    self._coco_cat_id_to_class = dict(zip(self._coco_cat_ids, [c['name'] for c in cats]))

    self._ind_to_coco_cat_id = dict(zip(xrange(self.num_classes),self._coco_cat_ids))
    self._coco_cat_id_to_ind = dict(zip(self._coco_cat_ids, xrange(self.num_classes)))

    imgs = self._coco_d['images']
    anns = self._coco_d['annotations']
    assert(len(imgs)==len(anns))
    print("There are %d images and %d annotations in total"%(len(imgs)),len(anns))
    #TODOTODOTODOTODOTODOTODOTODO
    #TODOTODOTODOTODOTODOTODOTODO
    self._img_paths = []
    self._ann_paths = []
    counter=0
    for ann in anns:
        img_path = osp.join(img_folder,ann['file_name'].replace('png','jpg'))
        ann_path = osp.join(segmentations_folder,img['file_name'])
        if osp.exists(img_path) and osp.exists(ann_path):
            counter+=1
            self._img_paths.append(img_path)
            self._ann_paths.append(ann_path)
            #print('file seen %d/%d'%(counter,len(imgs)),end="")
        else:
            print('Please check that image path: %s and annotation path: %s both exist'%(img_path,ann_path))
            print(counter)
    print('There are %d image and annotation pairs in total'%(counter))

    
    self._ps_db = None
    self._ps_db_handler = self.default_ps_db_handler

  def _get_coco_image_dir(self,split,year):
    coco_img_dir = osp.join(cfg.DATA_DIR,'coco','images')
    return osp.join(coco_img_dir,'%s%d'%(split,year))

  def _get_coco_seg_dir(self,split,year):
    coco_anns_dir = osp.join(cfg.DATA_DIR,'coco','annotations')
    return osp.join(coco_anns_dir,'panoptic_%s%d'%(split,year))

  def _get_coco_anns_json(self,split,year):
    coco_anns_dir = osp.join(cfg.DATA_DIR,'coco','annotations')
    return osp.join(coco_anns_dir,'panoptic_%s%d.json'%(split,year))


  def get_image_path_at(self,idx):
    return self._img_paths[idx]

  def get_label_path_at(self,idx):
    #_ann_paths and _img_paths are already aligned, get from <idx> directly.
    return self._ann_paths[idx]




