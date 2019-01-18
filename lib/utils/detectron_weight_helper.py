"""Helper functions for loading pretrained weights from Detectron pickle files
"""
import pickle
import re
import torch


def load_detectron_weight(net, detectron_weight_file, force_load_all = True, mapped_names_need_exist = False, raise_shape_mismatch_error=False):
    '''
    Args:
    force_load_all: if True, all param names should be in the mapping list.
    '''
    name_mapping, orphan_in_detectron = net.detectron_weight_mapping
    
    with open(detectron_weight_file, 'rb') as fp:
        src_blobs = pickle.load(fp, encoding='latin1')
    if 'blobs' in src_blobs:
        src_blobs = src_blobs['blobs']

    params = net.state_dict()
    not_in_mapping_list = []
    mapped_not_in_src_blobs = []
    src_blobs_keys = set(src_blobs.keys())
    for p_name, p_tensor in params.items():
        if not force_load_all:
            if p_name not in name_mapping:
                not_in_mapping_list.append(p_name)                
                continue
        d_name = name_mapping[p_name]
        if isinstance(d_name, str):  # maybe str, None or True
            if not mapped_names_need_exist and d_name not in src_blobs_keys:
                mapped_not_in_src_blobs.append('%s->%s'%(d_name, p_name))
                continue
            src_t = torch.Tensor(src_blobs[d_name])
            src_blobs_keys = src_blobs_keys-set([d_name])
            if not src_t.shape==p_tensor.shape:                
                if raise_shape_mismatch_error:
                    raise ValueError('detectron weight shape mis-match:%s'%(d_name))
                else:
                    print('detectron weight shape mis-match:%s'%(d_name))
                    print(src_t.shape,'->',p_tensor.shape)
            else:
                p_tensor.copy_(src_t)
    print('Following model weights are not in detectron mapping list:', not_in_mapping_list)
    print('Following source blob weights are not used:', src_blobs_keys)
    print('Following source blob weights not exist for model weights:', mapped_not_in_src_blobs)

def resnet_weights_name_pattern():
    pattern = re.compile(r"conv1_w|conv1_gn_[sb]|res_conv1_.+|res\d+_\d+_.+")
    return pattern


if __name__ == '__main__':
    """Testing"""
    from pprint import pprint
    import sys
    sys.path.insert(0, '..')
    from modeling.model_builder import Generalized_RCNN
    from core.config import cfg, cfg_from_file

    cfg.MODEL.NUM_CLASSES = 81
    cfg_from_file('../../cfgs/res50_mask.yml')
    net = Generalized_RCNN()

    # pprint(list(net.state_dict().keys()), width=1)

    mapping, orphans = net.detectron_weight_mapping
    state_dict = net.state_dict()

    for k in mapping.keys():
        assert k in state_dict, '%s' % k

    rest = set(state_dict.keys()) - set(mapping.keys())
    assert len(rest) == 0
