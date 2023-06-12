#!/bin/bash
# 加载模块
module load anaconda/2021.05
module load cuda/11.1
module load gcc/7.3

# 激活环境
source activate mmdet

# 切换路径
cd run/openmmlab/homework-3


python tools/featmap_vis_demo.py test/ config/rtm-det_100e.py work_dirs/rtm-det_100e/best_coco/bbox_mAP_epoch_96.pth --out-dir output-vis/backbone/ --target-layers backbone --channel-reduction squeeze_mean &
python tools/featmap_vis_demo.py test/ config/rtm-det_100e.py work_dirs/rtm-det_100e/best_coco/bbox_mAP_epoch_96.pth --out-dir output-vis/neck/ --target-layers neck --channel-reduction squeeze_mean &
python tools/boxam_vis_demo.py test/ config/rtmdet_100e.py work_dirs/rtm-det_100e/best_coco/bbox_mAP_epoch_96.pth --out-dir output-vis/grad-cam/neck-2/ --target-layer neck.out_convs[2] &
python tools/boxam_vis_demo.py test/ config/rtmdet_100e.py work_dirs/rtm-det_100e/best_coco/bbox_mAP_epoch_96.pth --out-dir output-vis/grad-cam/neck-0/ --target-layer neck.out_convs[0] 