import numpy as np
import cv2

from core.config import cfg
import utils.blob as blob_utils
import roi_data.rpn
from vos_utils.flow_util import readFlowFile as flo_reader

def get_minibatch_blob_names(is_training=True):
    """Return blob names in the order in which they are read by the data loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names(is_training=is_training)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=is_training
        )
    return blob_names


def get_minibatch(roidb, target_scale = None):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}

    # Get the input image blob
    if cfg.MODEL.LOAD_FLOW_FILE:
        im_blob, im_scales, flo_blob = _get_image_blob(roidb, target_scale)
        blobs['data_flow'] = flo_blob
    else:
        im_blob, im_scales = _get_image_blob(roidb, target_scale)
    blobs['data'] = im_blob
    
    if cfg.RPN.RPN_ON:
        # RPN-only or end-to-end Faster/Mask R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    elif cfg.RETINANET.RETINANET_ON:
        raise NotImplementedError
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid

def _flo_to_blob(flo, target_size):
    """
    """
    flo, flo_scale = blob_utils.prep_im_for_blob(flo, (0, 0), [target_size], cfg.TRAIN.MAX_SIZE)
    flo = flo[0]*flo_scale # scale the value.
    flo_scale = flo_scale[0]

    max_shape = blob_utils.get_max_shape([flo.shape[:2]])
    blob = np.zeros((1, max_shape[0], max_shape[1], 2), dtype=np.float32)
    blob[0, 0:flo.shape[0], 0:flo.shape[1], :] = flo
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def _get_image_blob(roidb, target_scale = None):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    if target_scale is None:
        scale_inds = np.random.randint(
            0, high=len(cfg.TRAIN.SCALES), size=num_images)

    processed_ims = []
    im_scales = []
    flo_list = []
    for i in range(num_images):
        im = cv2.imread(roidb[i]['image'])
        assert im is not None, \
            'Failed to read image \'{}\''.format(roidb[i]['image'])
        # If NOT using opencv to read in images, uncomment following lines
        # if len(im.shape) == 2:
        #     im = im[:, :, np.newaxis]
        #     im = np.concatenate((im, im, im), axis=2)
        # # flip the channel, since the original one using cv2
        # # rgb -> bgr
        # im = im[:, :, ::-1]
        if roidb[i]['flipped']:
            im = im[:, ::-1, :]
        if target_scale is None:
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
        else:
            target_size = target_scale
        im, im_scale = blob_utils.prep_im_for_blob(
            im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
        im_scales.append(im_scale[0])
        processed_ims.append(im[0])

        if cfg.MODEL.LOAD_FLOW_FILE:
            # prepare flow
            if roidb[i]['flow'] is not None:
                flo = flo_reader.read(roidb[i]['flow'])
                assert flo is not None, \
                    'Failed to read flow \'{}\''.format(roidb[i]['flow'])
                flo =  _flo_to_blob(flo, target_size)
                flo_list.append(flo)
            else:
                flo_list.append(None)

    # Create a blob to hold the input images [n, c, h, w]
    blob = blob_utils.im_list_to_blob(processed_ims)

    if cfg.MODEL.LOAD_FLOW_FILE:
        return blob, im_scales, flo_list
    else:
        return blob, im_scales
