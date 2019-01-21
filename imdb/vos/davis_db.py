from .vos_imdb import vos_imdb
import sys
import os
from os import path as osp
from PIL import Image
from matplotlib import pyplot as plt
from .config import cfg
import cv2
import numpy as np

davis_api_home = osp.join(cfg.DAVIS.HOME,'python','lib')
if not davis_api_home in sys.path:
  sys.path.append(davis_api_home)

from davis import cfg as cfg_davis
from davis import io,DAVISLoader,phase  

dataset_lib = osp.abspath('../../lib/')
if not dataset_lib in sys.path:
  sys.path.append(dataset_lib)
  
from datasets.dataset_catalog import ANN_FN
from datasets.dataset_catalog import DATASETS
if not cfg.COCO_API_HOME in sys.path:
  sys.path.append(cfg.COCO_API_HOME)

from pycocotools import mask as COCOmask
from pycocotools.coco import COCO

splits = ['train','val','trainval','test-dev']

def binary_mask_to_rle(binary_mask):
  rle = {'counts': [], 'size': list(binary_mask.shape)}
  counts = rle.get('counts')
  for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
    if i == 0 and value == 1:
        counts.append(0)
    counts.append(len(list(elements)))
  return rle

class DAVIS_imdb(vos_imdb):
  def __init__(self,db_name="DAVIS", split = 'train', cls_mapper = None):
    '''
    Args:
    cls_mapper(dict type): VOS dataset only provides instance id label or class label that
    is not consistent with the object detection model. As our work is to provide object 
    detection model with the ability for VOS task, so object label is provided by the
    prediction of object detection model. The prediction is provided by label_mapper.
    If set None, no class is assigned. Otherwise, category_id = cls_mapper[instance_id].
    
    For seq_idx, instance_idx, its class label can be got by "label_mapper[seq_idx][instance_idx]".
    
    As some objects may be predicted as background, we choose the class with highest probability 
    among non-background classes to be its class label.
    '''
    super().__init__(db_name)
    self.split = split
    if split is not None:
      if split not in splits:
        raise ValueError('split not recognizable')
      if split=='train':
        self.phase = phase.TRAIN
      elif split=='val':
        self.phase = phase.VAL
      elif split=='trainval':
        self.phase = phase.TRAINVAL
      elif split=='test-dev':
        self.phase = phase.TESTDEV
      else:
        raise ValueError('split not recognizable')
      if cfg_davis.PHASE!=self.phase:
        print('phase changed from %s to %s'%(cfg_davis.PHASE.value,self.phase.value))
        cfg_davis.PHASE = self.phase
    print('year:',cfg_davis.YEAR)
    print('phase:',cfg_davis.PHASE.value)
    self.db = DAVISLoader(year=cfg_davis.YEAR,phase=cfg_davis.PHASE)
    self.seq_idx = 0
    self.cls_mapper = None
    if cls_mapper is not None:
      assert(isinstance(cls_mapper, dict))
      self.cls_mapper = cls_mapper
    # Here we adopt COCO classes.
    
    self.COCO = COCO(DATASETS['coco_2017_train'][ANN_FN])
    category_ids = self.COCO.getCatIds()
    categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
    self.category_to_id_map = dict(zip(categories, category_ids))
    self.classes = ['__background__'] + categories + ['__unknown__']
    self.num_classes = len(self.classes)
    self.json_category_id_to_contiguous_id = {
        v: i + 1
        for i, v in enumerate(self.COCO.getCatIds())
    }
    self.number_of_instance_ids = 0
    self.global_instance_id_start_of_seq = np.zeros(self.get_num_sequence(),dtype=np.uint8)
    self.instance_number_of_seq = np.zeros(self.get_num_sequence(),dtype=np.uint8)
    self.set_global_instance_id_start()

  @property
  def valid_cached_keys(self):
      """ Can load following key-ed values from the cached roidb file

      'image'(image path) and 'flipped' values are already filled on _prep_roidb_entry,
      so we don't need to overwrite it again.
      """
      keys = ['boxes', 'segms', 'gt_classes', 'instance_id', 'global_instance_id', 'seg_areas', 'gt_overlaps','gt_overlaps_id', 'is_crowd', 'box_to_gt_ind_map']
      return keys

  def get_num_sequence(self):
    return len(self.db.sequences)

  def set_to_sequence(self,seq_idx):
    assert(seq_idx>=0 and seq_idx<self.get_num_sequence())
    self.seq_idx = seq_idx
  
  def get_current_seq_name(self):
    return self.db.sequences[self.seq_idx].name
  
  def get_current_seq(self):
    return self.db.sequences[self.seq_idx]

  def get_current_seq_index(self):
    return self.seq_idx
    
  def get_current_seq_length(self):
    return len(self.get_current_seq().files)

  def get_image(self, idx):
    seq = self.get_current_seq()
    return np.array(Image.open(seq.files[idx]))
    
  def get_image_cv2(self,idx):
    seq = self.get_current_seq()
    return cv2.imread(seq.files[idx])

  def get_gt_with_color(self, idx):
    return cfg_davis.palette[self.db.annotations[self.seq_idx][idx]][...,[2,1,0]]

  def get_gt(self,idx):
    return self.db.annotations[self.seq_idx][idx]

  def get_bboxes(self,idx):
    gt = self.get_gt(idx)
    vals = np.unique(gt)
    boxes = []
    for val in vals:
      #it is background when val==0
      if val !=0:
        obj={}
        mask = np.array(gt==val,dtype=np.uint8)
        #make sure gt==val is converted to value in 0 and 1.
        assert(len(set(mask.reshape(-1))-{0,1})==0)
        x,y,w,h = cv2.boundingRect(mask)
        boxes.append([x,y,w,h])
    return boxes

  def set_global_instance_id_start(self):
      #start from 1. 0 is reserved for background.      
      accumulate = 1
      self.global_instance_id_start_of_seq[0] = accumulate
      for seq_idx in range(self.get_num_sequence()-1):
        if self.instance_number_of_seq[seq_idx] == 0:
          self.set_to_sequence(seq_idx)
          #get gt from first frame.
          gt = self.get_gt(idx=0)
          #len(np.unique(gt))-1 as there is background in vals.
          self.instance_number_of_seq[seq_idx] = len(np.unique(gt))-1
        accumulate += self.instance_number_of_seq[seq_idx]
        self.global_instance_id_start_of_seq[seq_idx+1] = accumulate
      print('instance_number_of_seq:',self.instance_number_of_seq)
      print('global_instance_id_start_of_seq:',self.global_instance_id_start_of_seq)
      self.number_of_instance_ids = accumulate+self.instance_number_of_seq[self.get_num_sequence()-1]
      print('Total global instance id number(include background):%d'%(self.number_of_instance_ids))
      
      
  def set_number_of_instance(self, seq_idx, num_instances):
      self.instance_number_of_seq[seq_idx] = num_instances

  def visualize_blended_image_label(self,idx,w1=0.5,w2=0.5):
    '''
    Args:
    w1: weight for Image
    w2: weight for label
    '''
    img = self.get_image(idx)
    gt = self.get_gt_with_color(idx)
    img_cpy = copy(img)
    mask = np.array(np.all(gt[:,:,:]==0,axis=2,keepdims=True),dtype=np.uint8)
    unmasked_img = np.array(img_cpy*mask,dtype=np.uint8)
    mask_img = img-unmasked_img
    blend = unmasked_img+np.array(mask_img*w1+gt*w2,dtype=np.uint8)
    plt.imshow(blend)
    plt.show()
    return blend

  def get_roidb_at_idx_from_sequence(self,idx):
    roidb = {}
    seq = self.get_current_seq()
    gt = self.get_gt(idx)
    roidb['image'] = seq.files[idx]
    roidb['height'] = gt.shape[0]
    roidb['width'] = gt.shape[1]
    roidb['seq_idx'] = self.get_current_seq_index()
    roidb['idx'] = idx
    return roidb
   
  def prepare_roi_db(self, roidb):
    for entry in roidb:
      self._prep_roidb_entry(entry)
    if self.split in ['train','val','trainval']:
      for entry in roidb:
        self._add_gt_annotations(entry)

  def _prep_roidb_entry(self, entry):
    """Adds empty metadata fields to an roidb entry."""
    # Reference back to the parent dataset
    entry['dataset'] = self
    im_path = entry['image']
    assert os.path.exists(im_path), 'Image \'{}\' not found'.format(im_path)
    entry['flipped'] = False
    entry['has_visible_keypoints'] = False
    # Empty placeholders
    entry['boxes'] = np.empty((0, 4), dtype=np.float32)
    entry['segms'] = []
    entry['gt_classes'] = np.empty((0), dtype=np.int32)
    entry['global_instance_id'] = np.empty((0), dtype=np.int32)
    entry['instance_id'] = np.empty((0), dtype=np.int32)
    
    entry['seg_areas'] = np.empty((0), dtype=np.float32)
    entry['gt_overlaps'] = scipy.sparse.csr_matrix(
        np.empty((0, self.num_classes), dtype=np.float32)
    )
    entry['gt_overlaps_id'] = scipy.sparse.csr_matrix(
        np.empty((0, self.number_of_instance_ids), dtype=np.float32)
    )
    entry['is_crowd'] = np.empty((0), dtype=np.bool)
    # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
    # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
    entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)


  def _add_gt_annotations(self, entry):
    """Add ground truth annotation metadata to an roidb entry.
    """
    seq_idx = entry['seq_idx']
    idx = entry['idx']
    self.set_to_sequence(seq_idx)
    #get gt image.
    gt = self.get_gt(idx)
    vals = np.unique(gt)
    
    objs = []
    for val in vals:
      #it is background when val==0
      if val !=0:
        obj={}
        mask = np.array(gt==val,dtype=np.uint8)
        #make sure gt==val is converted to value in 0 and 1.
        assert(len(set(mask.reshape(-1))-{0,1})==0)
        x,y,w,h = cv2.boundingRect(mask)

        obj['segmentation'] = binary_mask_to_rle(np.np.asfortranarray(mask))
        obj['area'] = np.sum(mask)
        obj['iscrowd'] = 0
        obj['bbox'] = x,y,w,h
        if self.cls_mapper is not None:
          #set category id by cls_mapper.
          obj['category_id'] = self.cls_mapper[val]
        else:
          obj['category_id'] = None
        obj['instance_id'] = val
        assert(self.global_instance_id_start_of_seq[seq_idx]!=0)
        # val-1 to remove background.
        obj['global_instance_id'] = self.global_instance_id_start_of_seq[seq_idx]+val-1
        objs.append(obj)

    # Sanitize bboxes -- some are invalid
    valid_objs = []
    valid_segms = []
    width = entry['width']
    height = entry['height']
    for obj in objs:
      # crowd regions are RLE encoded and stored as dicts
      if isinstance(obj['segmentation'], list):
        # Valid polygons have >= 3 points, so require >= 6 coordinates
        obj['segmentation'] = [
          p for p in obj['segmentation'] if len(p) >= 6
        ]

      # Convert form (x1, y1, w, h) to (x1, y1, x2, y2)
      x1, y1, x2, y2 = box_utils.xywh_to_xyxy(obj['bbox'])
      x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
          x1, y1, x2, y2, height, width
      )
      # Require non-zero seg area and more than 1x1 box size
      if obj['area'] > 0 and x2 > x1 and y2 > y1:
        obj['clean_bbox'] = [x1, y1, x2, y2]
        valid_objs.append(obj)
        valid_segms.append(obj['segmentation'])
    num_valid_objs = len(valid_objs)

    boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
    gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
    instance_id = np.zeros((num_valid_objs), dtype=entry['instance_id'].dtype)
    global_instance_id = np.zeros((num_valid_objs), dtype=entry['global_instance_id'].dtype)
    gt_overlaps = np.zeros(
      (num_valid_objs, self.num_classes),
      dtype=entry['gt_overlaps'].dtype
    )
    gt_overlaps_id = np.zeros(
      (num_valid_objs, self.number_of_instance_ids),
      dtype=entry['gt_overlaps_id'].dtype
    )
    seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
    is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
    box_to_gt_ind_map = np.zeros(
      (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype
    )

    im_has_visible_keypoints = False
    for ix, obj in enumerate(valid_objs):
      if 'category_id' is not None:
        cls = self.json_category_id_to_contiguous_id[obj['category_id']]
      else:
        #if no category_id specified, use background instead. index is 'self.num_classes-1'
        cls = self.num_classes-1
      boxes[ix, :] = obj['clean_bbox']
      gt_classes[ix] = cls
      instance_id[ix] = obj['instance_id']
      global_instance_id[ix] = obj['global_instance_id']
      seg_areas[ix] = obj['area']
      is_crowd[ix] = obj['iscrowd']
      box_to_gt_ind_map[ix] = ix
      if obj['iscrowd']:
        # Set overlap to -1 for all classes for crowd objects
        # so they will be excluded during training
        gt_overlaps[ix, :] = -1.0
        gt_overlaps_id[ix,:] = -1.0
      else:
        gt_overlaps[ix, cls] = 1.0
        gt_overlaps_id[ix, global_instance_id[ix]] = 1.0
    entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
    entry['segms'].extend(valid_segms)
    # To match the original implementation:
    # entry['boxes'] = np.append(
    #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
    entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
    entry['instance_id'] = np.append(entry['instance_id'], instance_id)
    entry['global_instance_id'] = np.append(entry['global_instance_id'], global_instance_id)
    entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
    entry['gt_overlaps'] = np.append(
      entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
    )
    entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
    
    entry['gt_overlaps_id'] = np.append(
      entry['gt_overlaps_id'].toarray(), gt_overlaps_id, axis=0
    )
    entry['gt_overlaps_id'] = scipy.sparse.csr_matrix(entry['gt_overlaps_id'])    
    entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
    entry['box_to_gt_ind_map'] = np.append(
      entry['box_to_gt_ind_map'], box_to_gt_ind_map
    )

  def _add_gt_from_cache(self, roidb, cache_filepath):
    """Add ground truth annotation metadata from cached file."""
    logger.info('Loading cached gt_roidb from %s', cache_filepath)
    with open(cache_filepath, 'rb') as fp:
      cached_roidb = pickle.load(fp)

    assert len(roidb) == len(cached_roidb)

    for entry, cached_entry in zip(roidb, cached_roidb):
      values = [cached_entry[key] for key in self.valid_cached_keys]
      boxes, segms, gt_classes, instance_id, global_instance_id, seg_areas, gt_overlaps, gt_overlaps_id, is_crowd, \
        box_to_gt_ind_map = values[:10]
      entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
      entry['segms'].extend(segms)
      # To match the original implementation:
      # entry['boxes'] = np.append(
      #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
      entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
      entry['instance_id'] = np.append(entry['instance_id'], instance_id)
      entry['global_instance_id'] = np.append(entry['global_instance_id'], global_instance_id)
      entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
      entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)
      entry['gt_overlaps_id'] = scipy.sparse.csr_matrix(gt_overlaps_id)
      entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
      entry['box_to_gt_ind_map'] = np.append(
        entry['box_to_gt_ind_map'], box_to_gt_ind_map
      )

  def get_roidb_from_seq_idx_sequence(self, seq_idx):
    roidb = []
    
    self.debug_timer.tic()
    self.set_to_sequence(seq_idx)
    seq_len = self.get_current_seq_length()
    for idx in range(seq_len):
      roidb.append(self.get_roidb_at_idx_from_sequence(idx))

    for entry in roidb:
      self._prep_roidb_entry(entry)
 
    # Include ground-truth object annotations
    if not osp.isdir(cfg.CACHE_DIR):
      os.mkdirs(cfg.CACHE_DIR)
      assert(osp.isdir(cfg.CACHE_DIR))
    cache_filepath = os.path.join(cfg.CACHE_DIR, self.name+'_%d_sequence_roidb.pkl'%(seq_idx))
    if os.path.exists(cache_filepath) and not cfg.DEBUG:
        self._add_gt_from_cache(roidb, cache_filepath)
        logger.debug(
            '_add_gt_from_cache took {:.3f}s'.
            format(self.debug_timer.toc(average=False))
        )
    else:
      if self.split in ['train','val','trainval']:
        for entry in roidb:
          self._add_gt_annotations(entry)
        logger.debug(
            '_add_gt_annotations took {:.3f}s'.
            format(self.debug_timer.toc(average=False))
        )
        if not cfg.DEBUG:
            with open(cache_filepath, 'wb') as fp:
                pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
            logger.info('Cache ground truth roidb to %s', cache_filepath)
    return roidb

    
  def get_roidb_from_all_sequences(self):
    roidb = []
    self.debug_timer.tic()
    for seq_idx in range(self.get_num_sequence()):
      self.set_to_sequence(seq_idx)
      seq_len = self.get_current_seq_length()
      for idx in range(seq_len):
        roidb.append(self.get_roidb_at_idx_from_sequence(idx))
    
    for entry in roidb:
      self._prep_roidb_entry(entry)
    # Include ground-truth object annotations
    if not osp.isdir(cfg.CACHE_DIR):
        os.mkdirs(cfg.CACHE_DIR)
        assert(osp.isdir(cfg.CACHE_DIR))
    cache_filepath = os.path.join(cfg.CACHE_DIR, self.name+'_all_sequences_roidb.pkl')
    if os.path.exists(cache_filepath) and not cfg.DEBUG:
        self._add_gt_from_cache(roidb, cache_filepath)
        logger.debug(
            '_add_gt_from_cache took {:.3f}s'.
            format(self.debug_timer.toc(average=False))
        )
    logger.debug(
        '_add_gt_annotations took {:.3f}s'.
        format(self.debug_timer.toc(average=False))
    )
    else:
      if self.split in ['train','val','trainval']:
        for entry in roidb:
          self._add_gt_annotations(entry)
        logger.debug(
            '_add_gt_annotations took {:.3f}s'.
            format(self.debug_timer.toc(average=False))
        )      
      if not cfg.DEBUG:
          with open(cache_filepath, 'wb') as fp:
              pickle.dump(roidb, fp, pickle.HIGHEST_PROTOCOL)
          logger.info('Cache ground truth roidb to %s', cache_filepath)
    return roidb
  
  def _create_db_from_label(self):
    pass



def add_proposals(roidb, rois, scales, crowd_thresh):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    if crowd_thresh > 0:
        _filter_crowd_proposals(roidb, crowd_thresh)
    _add_class_assignments(roidb)

def _merge_proposal_boxes_into_roidb(roidb, box_list):
    """Add proposal boxes to each roidb entry."""
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype
        )
        
        gt_overlaps_id = np.zeros(
            (num_boxes, entry['gt_overlaps_id'].shape[1]),
            dtype=entry['gt_overlaps_id'].dtype
        )
                
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype
        )

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            global_instance_id = entry['global_instance_id'][gt_inds]
            
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False)
            )
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            gt_overlaps_id[I,global_instance_id[argmaxes[I]]] = maxes[I]
            
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0
        )
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype)
        )
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype)
        )
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0
        )
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        
        entry['gt_overlaps_id'] = np.append(
            entry['gt_overlaps_id'].toarray(), gt_overlaps_id, axis=0
        )
        entry['gt_overlaps_id'] = scipy.sparse.csr_matrix(entry['gt_overlaps_id'])
        
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype)
        )
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False
            )
        )

def _add_class_assignments(self, roidb):
    """Compute object category assignment for each box associated with each
    roidb entry.
    """
    for entry in roidb:
      gt_overlaps = entry['gt_overlaps'].toarray()
      # max overlap with gt over classes (columns)
      max_overlaps = gt_overlaps.max(axis=1)
      # gt class that had the max overlap
      max_classes = gt_overlaps.argmax(axis=1)
      entry['max_classes'] = max_classes
      entry['max_overlaps'] = max_overlaps
      
      gt_overlaps_id = entry['gt_overlaps_id'].toarray()
      # max overlap with gt over classes (columns)
      max_overlaps_id = gt_overlaps_id.max(axis=1)
      # gt class that had the max overlap
      max_global_id = gt_overlaps_id.argmax(axis=1)
      entry['max_global_id'] = max_global_id
      entry['max_overlaps_id'] = max_overlaps_id
      
      # sanity checks
      # if max overlap is 0, the class must be background (class 0)
      zero_inds = np.where(max_overlaps == 0)[0]
      assert all(max_classes[zero_inds] == 0)
      # if max overlap > 0, the class must be a fg class (not class 0)
      nonzero_inds = np.where(max_overlaps > 0)[0]
      assert all(max_classes[nonzero_inds] != 0)
      
      # sanity checks
      # if max overlap id is 0, the id must be background (id 0)
      zero_inds = np.where(max_overlaps_id == 0)[0]
      assert all(max_global_id[zero_inds] == 0)
      # if max overlap id > 0, the id must be a fg id (not id 0)
      nonzero_inds = np.where(max_overlaps_id > 0)[0]
      assert all(max_global_id[nonzero_inds] != 0)

def _sort_proposals(proposals, id_field):
  """Sort proposals by the specified id field."""
  order = np.argsort(proposals[id_field])
  fields_to_sort = ['boxes', id_field, 'scores']
  for k in fields_to_sort:
      proposals[k] = [proposals[k][i] for i in order]
