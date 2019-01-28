import math
import numpy as np
import numpy.random as npr

import torch
import torch.utils.data as data
import torch.utils.data.sampler as torch_sampler
from torch.utils.data.dataloader import default_collate
from torch._six import int_classes as _int_classes

from core.config import cfg
from roi_data.minibatch import get_minibatch
import utils.blob as blob_utils
from random import randint as randi
# from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

def addPath(path):
  if path not in sys.path:
    sys.path.append(path)
lib_path = osp.abspath(osp.join(osp.dirname(__file__),'../../lib/'))
addPath(lib_path)

class RoiDataLoader(data.Dataset):
    def __init__(self, roidb, num_classes, training=True):
        self._roidb = roidb
        self._num_classes = num_classes
        self.training = training
        self.DATA_SIZE = len(self._roidb)

    def __getitem__(self, index):  
        single_db = [self._roidb[index]]
        blobs, valid = get_minibatch(single_db)
        #TODO: Check if minibatch is valid ? If not, abandon it.
        # Need to change _worker_loop in torch.utils.data.dataloader.py.
        # Squeeze batch dim
        for key in blobs:
            if key != 'roidb':
                blobs[key] = blobs[key].squeeze(axis=0)

        blobs['roidb'] = blob_utils.serialize(blobs['roidb'])  # CHECK: maybe we can serialize in collate_fn
        return blobs
        
    def __len__(self):
        return self.DATA_SIZE


class MinibatchSampler(torch_sampler.Sampler):
    def __init__(self, seq_num, seq_start_end):
        self.seq_num = seq_num
        self.seq_start_end = seq_start_end
        self.num_data = seq_num

    def __iter__(self):
        rand_perm = npr.permutation(self.seq_num)
        shuffled_seq_start_end = self.seq_start_end[rand_perm,:]        
        #randomly sample from sequence.
        idx_list = []
        for idx in range(self.seq_num):
            cur_se = shuffled_seq_start_end[idx,:]
            assert(cur_se[1] > cur_se[0])
            length = cur_se[1] - cur_se[0]
            if length<cfg.SEQUENCE_LENGTH:
                raise ValueError('length of the sequence is shorter than cfg.SEQUENCE_LENGTH.')
            else:
                id_start = randi(0, length-cfg.SEQUENCE_LENGTH)
                id_end = id_start+cfg.SEQUENCE_LENGTH
                idx_list.append(list(range(id_start, id_end)))        
        return iter(idx_list)

    def __len__(self):
        return self.seq_num


class BatchSampler(torch_sampler.BatchSampler):
    r"""Wraps another sampler to yield a mini-batch of indices.
    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    Example:
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(range(10), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, torch_sampler.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)  # Difference: batch.append(int(idx))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



def collate_minibatch(list_of_blobs):
    """Stack samples seperately and return a list of minibatches
    A batch contains NUM_GPUS minibatches and image size in different minibatch may be different.
    Hence, we need to stack smaples from each minibatch seperately.
    """
    Batch = {key: [] for key in list_of_blobs[0]}
    # Because roidb consists of entries of variable length, it can't be batch into a tensor.
    # So we keep roidb in the type of "list of ndarray".
    list_of_roidb = [blobs.pop('roidb') for blobs in list_of_blobs]
    for i in range(0, len(list_of_blobs), cfg.TRAIN.IMS_PER_BATCH):
        mini_list = list_of_blobs[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        # Pad image data
        mini_list = pad_image_data(mini_list)
        minibatch = default_collate(mini_list)
        minibatch['roidb'] = list_of_roidb[i:(i + cfg.TRAIN.IMS_PER_BATCH)]
        for key in minibatch:
            Batch[key].append(minibatch[key])
    return Batch


def pad_image_data(list_of_blobs):
    max_shape = blob_utils.get_max_shape([blobs['data'].shape[1:] for blobs in list_of_blobs])
    output_list = []
    for blobs in list_of_blobs:
        data_padded = np.zeros((3, max_shape[0], max_shape[1]), dtype=np.float32)
        _, h, w = blobs['data'].shape
        data_padded[:, :h, :w] = blobs['data']
        blobs['data'] = data_padded
        output_list.append(blobs)
    return output_list
