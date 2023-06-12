#!/bin/bash
# 加载模块
module load anaconda/2021.05
module load cuda/11.1
module load gcc/7.3

# 激活环境
source activate mmdet

# 切换路径
cd run/openmmlab/homework-3


python tools/train.py config/rtm-det_100e.py