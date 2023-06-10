## 题目：基于 ResNet50 的水果分类

## 背景：使用基于卷积的深度神经网络 ResNet50 对 30 种水果进行分类

## 任务

1. 划分训练集和验证集
2. 按照 MMPreTrain CustomDataset 格式组织训练集和验证集
3. 使用 MMPreTrain 算法库，编写配置文件，正确加载预训练模型
4. 在水果数据集上进行微调训练
5. 使用 MMPreTrain 的 ImageClassificationInferencer 接口，对网络水果图像，或自己拍摄的水果图像，使用训练好的模型进行分类
6. 需提交的验证集评估指标（不能低于 60%）
> ResNet-50

![img](resource/243638043-21dcc64f-8537-4a5a-9938-5bbf2cc47a0e.png)

## 作业数据集下载：
链接：https://pan.baidu.com/s/1YgoU1M_v7ridtXB9xxbA1Q
提取码：52m9

## 课程中猫狗数据集下载地址：
https://download.openmmlab.com/mmclassification/dataset/cats_dogs_dataset.tar

# 作业内容

0. [数据集划分](dataset_handle/split_dataset.py)
1. [配置文件](config/resnet50_1xb256_coslr-100e_fruit-30.py)
2. [运行日志](log/20230609_172618/20230609_172618.log)

## 运行指标

```bash
2023/06/09 17:44:29 - mmengine - INFO - Epoch(val) [94][4/4]    accuracy/top1: 92.9054  accuracy/top5: 98.6487  data_time: 0.1743  time: 0.3092
2023/06/09 17:44:29 - mmengine - INFO - The previous best checkpoint /data/run01/scz0brk/openmmlab/mmpretrain/work_dir/resnet50/resnet50_1xb256_coslr-100e_fruit-30/best_accuracy_top1_epoch_91.pth is removed
2023/06/09 17:44:29 - mmengine - INFO - The best checkpoint with 92.9054 accuracy/top1 at 94 epoch is saved to best_accuracy_top1_epoch_94.pth.
```

## 推理结果

### 混淆矩阵

![混淆矩阵](result/fruit_matrix.png)

### 推理结果

![草莓](result/草莓.png)
![青苹果](result/青苹果.png)
