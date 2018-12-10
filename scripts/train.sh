#!/usr/bin/env bash

python ./tools/train_net_step.py --dataset coco2017 --cfg ./scripts/e2e_mask_rcnn_R-101-FPN_3x_gn.yaml --disp_interval 50 --nw 8 --o SGD --use_tfboard --load_ckpt "/media/yelyu/18339a64-762e-4258-a609-c0851cd8163e/YeLyu/Pytorch/Detectron.pytorch/Outputs/e2e_mask_rcnn_R-101-FPN_3x_gn/Dec05-17-06-32_UT143397_step/ckpt/model_step239999.pth"
