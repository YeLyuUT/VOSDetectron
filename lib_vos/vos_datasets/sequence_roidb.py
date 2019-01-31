"""Functions for common roidb manipulations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from os import path as osp
import sys
def addPath(path):
  if path not in sys.path:
    sys.path.append(path)
imdb_path = osp.abspath(osp.join(osp.dirname(__file__),'../../imdb/'))
addPath(imdb_path)
lib_path = osp.abspath(osp.join(osp.dirname(__file__),'../../lib/'))
addPath(lib_path)

import six
import logging
import numpy as np

import utils.boxes as box_utils
import utils.keypoints as keypoint_utils
import utils.segms as segm_utils
import utils.blob as blob_utils
from core.config import cfg

logger = logging.getLogger(__name__)

from vos.davis_db import DAVIS_imdb

def sequenced_roidb_for_training_from_db(dbs, proposal_files):    
    def get_roidb(db, proposal_file):
        roidb = None
        dataset_name = db._dbname.lower()
        if 'davis' in dataset_name:
          ds = db
          roidbs = ds.get_separate_roidb_from_all_sequences(proposal_file=proposal_file)
        else:
          raise ValueError('Invalid dataset type.')
          
        if cfg.TRAIN.USE_FLIPPED:
            raise ValueError('cfg.TRAIN.USE_FLIPPED is True, but not implemented yet.')
            #logger.info('Appending horizontally-flipped training examples...')
            #extend_with_flipped_entries(roidb, ds)
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidbs

    if len(proposal_files) == 0:
        proposal_files = (None, ) * len(dbs)
    assert len(dbs) == len(proposal_files)
    #roidbs is list of roidbs.
    roidbss = [get_roidb(*args) for args in zip(dbs, proposal_files)]    
    roidbs = roidbss[0]
    for r in roidbss[1:]:
        roidbs.extend(r)
    
    #remove filter.
    #for i in range(len(roidbs)):
     # roidbs[i] = filter_for_training(roidbs[i])

    logger.info('Computing bounding-box regression targets...')
    for roidb in roidbs:
      add_bbox_regression_targets(roidb)
    logger.info('done')

    _compute_and_log_stats(roidbs)
    
    merged_roidb, seq_num, seq_start_end = _merge_roidbs(roidbs)
        
    return merged_roidb, seq_num, seq_start_end

def sequenced_roidb_for_training(dataset_names, proposal_files, use_local_id = False):
    """Load and concatenate roidbs for one or more datasets, along with optional
    object proposals. The roidb entries are then prepared for use in training,
    which involves caching certain types of metadata for each roidb entry.
    """
    def get_roidb(dataset_name, proposal_file):
        roidb = None
        dataset_name = dataset_name.lower()
        if 'davis' in dataset_name:
          name, split = dataset_name.split('_')
          #year = '2017', split = 'train'
          ds = DAVIS_imdb(db_name="DAVIS", split = split, cls_mapper = None, load_flow=cfg.MODEL.LOAD_FLOW_FILE, use_local_id = use_local_id)
          #roidb = ds.get_roidb_from_all_sequences()
          roidbs = ds.get_separate_roidb_from_all_sequences(proposal_file=proposal_file)
        else:
          raise ValueError('Invalid dataset type.')
          
        if cfg.TRAIN.USE_FLIPPED:
            raise ValueError('cfg.TRAIN.USE_FLIPPED is True, but not implemented yet.')
            #logger.info('Appending horizontally-flipped training examples...')
            #extend_with_flipped_entries(roidb, ds)
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidbs

    if isinstance(dataset_names, six.string_types):
        dataset_names = (dataset_names, )
    if isinstance(proposal_files, six.string_types):
        proposal_files = (proposal_files, )
    if len(proposal_files) == 0:
        proposal_files = (None, ) * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    #roidbs is list of roidbs.
    roidbss = [get_roidb(*args) for args in zip(dataset_names, proposal_files)]    
    roidbs = roidbss[0]
    for r in roidbss[1:]:
        roidbs.extend(r)
    
    #remove filter.
    #for i in range(len(roidbs)):
     # roidbs[i] = filter_for_training(roidbs[i])

    logger.info('Computing bounding-box regression targets...')
    for roidb in roidbs:
      add_bbox_regression_targets(roidb)
    logger.info('done')

    _compute_and_log_stats(roidbs)
    
    merged_roidb, seq_num, seq_start_end = _merge_roidbs(roidbs)
        
    return merged_roidb, seq_num, seq_start_end

def _merge_roidbs(roidbs):
    seq_num = len(roidbs)
    seq_start_end = np.zeros([seq_num,2], np.int32)
    seq_start = 0
    seq_end = 0
    merged_roidb = []
    for seq_idx in range(seq_num):
      roidb = roidbs[seq_idx]
      seq_start_end[seq_idx,0] = seq_start
      seq_end = seq_start+len(roidb)
      seq_start_end[seq_idx,1] = seq_end
      seq_start = seq_end
      merged_roidb.extend(roidb)
    return merged_roidb, seq_num, seq_start_end

def extend_with_flipped_entries(roidb, dataset):
    """Flip each entry in the given roidb and return a new roidb that is the
    concatenation of the original roidb and the flipped entries.

    "Flipping" an entry means that that image and associated metadata (e.g.,
    ground truth boxes and object proposals) are horizontally flipped.
    """
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        flipped_entry['segms'] = segm_utils.flip_segms(
            entry['segms'], entry['height'], entry['width']
        )
        if dataset.keypoints is not None:
            flipped_entry['gt_keypoints'] = keypoint_utils.flip_keypoints(
                dataset.keypoints, dataset.keypoint_flip_map,
                entry['gt_keypoints'], entry['width']
            )
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]              
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if cfg.MODEL.KEYPOINTS_ON:
            # If we're training for keypoints, exclude images with no keypoints
            valid = valid and entry['has_visible_keypoints']
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.format(num - num_after, num, num_after))
    return filtered_roidb


def rank_for_training(roidb):
    """Rank the roidb entries according to image aspect ration and mark for cropping
    for efficient batching if image is too long.

    Returns:
        ratio_list: ndarray, list of aspect ratios from small to large
        ratio_index: ndarray, list of roidb entry indices correspond to the ratios
    """
    RATIO_HI = cfg.TRAIN.ASPECT_HI  # largest ratio to preserve.
    RATIO_LO = cfg.TRAIN.ASPECT_LO  # smallest ratio to preserve.

    need_crop_cnt = 0

    ratio_list = []
    for entry in roidb:
        width = entry['width']
        height = entry['height']
        ratio = width / float(height)

        if cfg.TRAIN.ASPECT_CROPPING:
            if ratio > RATIO_HI:
                entry['need_crop'] = True
                ratio = RATIO_HI
                need_crop_cnt += 1
            elif ratio < RATIO_LO:
                entry['need_crop'] = True
                ratio = RATIO_LO
                need_crop_cnt += 1
            else:
                entry['need_crop'] = False
        else:
            entry['need_crop'] = False

        ratio_list.append(ratio)

    if cfg.TRAIN.ASPECT_CROPPING:
        logging.info('Number of entries that need to be cropped: %d. Ratio bound: [%.2f, %.2f]',
                     need_crop_cnt, RATIO_LO, RATIO_HI)
    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = _compute_targets(entry)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th)
    targets = np.zeros((rois.shape[0], 5), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _compute_and_log_stats(roidbs):
    classes = roidbs[0][0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)
    
    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for roidb in roidbs:
      for entry in roidb:
          gt_inds = np.where(
              (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
          gt_classes = entry['gt_classes'][gt_inds]
          gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.debug('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.debug(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.debug('-' * char_len)
    logger.debug(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))
