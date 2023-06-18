## 作业题目

### 背景：西瓜瓤、西瓜皮、西瓜籽像素级语义分割

**TO DO LIST：**

1. Labelme 标注语义分割数据集（子豪兄已经帮你完成了）
2. 划分训练集和测试集（子豪兄已经帮你完成了）
3. Labelme 标注转 Mask 灰度图格式（子豪兄已经帮你完成了）
4. 使用 MMSegmentation 算法库，撰写 config 配置文件，训练 PSPNet 语义分割算法
5. 提交测试集评估指标
6. 使用数据集之外的西瓜图片和视频进行预测，并存储展示预测的结果。
7. 训练 Segformer 语义分割算法，提交测试集评估指标
8. 西瓜瓤、西瓜籽数据集：

标注：同济子豪兄

![img](https://user-images.githubusercontent.com/129837368/245073269-598d8e55-62b0-438b-87c5-15fc6df9a365.png)

![img](https://user-images.githubusercontent.com/129837368/245073289-6d50954b-8b87-4a47-a54a-a55a720e30ac.png)

![img](https://user-images.githubusercontent.com/8240984/245413805-31a2e839-4cc6-419b-80d5-088bb4c4d755.png)

**数据集下载链接：**

- Labelme标注格式（没有划分训练集和测试集）：https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Labelme.zip

- Mask标注格式（已划分训练集和测试集）：https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Mask.zip

**需提交的测试集评估指标：（不能低于 baseline 指标的 50% ）**

- aAcc: 60.6200
- mIoU: 21.1400
- mAcc: 28.4600

**需要提交的其他文件**

1. 训练config
2. 训练日志log和输出文件夹work_dirs
3. 测试图片视频原图即推理后的图片视频
4. 评估指标，config路径地址，log地址，图片视频地址放在提交文件夹的readme.md文件中

## 准备数据集

```bash
wget https://zihao-openmmlab.obs.cn-east-3.myhuaweicloud.com/20230130-mmseg/dataset/watermelon/Watermelon87_Semantic_Seg_Mask.zip -P data/
unzip data/Watermelon87_Semantic_Seg_Mask.zip -d data/
```

训练集和验证集中各有一个张图片路径有问题：

```bash
mv img_dir/train/21746.1.jpg img_dir/train/21746.jpg
mv img_dir/val/01bd15599c606aa801201794e1fa30.jpg@1280w_1l_2o_100sh.jpg img_dir/val/01bd15599c606aa801201794e1fa30.jpg
```


## 准备配置文件

## 定义数据集类

在 `mmseg/datasets/` 目录下定义自己的数据集类 `HomeworkDataset.py`

```python
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class HomeworkDataset(BaseSegDataset):
    # 类别和对应的可视化配色
    METAINFO = {
        'classes':['background', 'red', 'green', 'white', 'seed-black', 'seed-white'],
        'palette':[[132,41,246], [228,193,110], [152,16,60], [58,221,254], [41,169,226], [155,155,155]]
    }
    
    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False, # 类别ID为0的类别是否需要除去
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
```

其中，[reduce_zero_label](https://mmsegmentation.readthedocs.io/zh_CN/latest/notes/faq.html?highlight=reduce_zero_label#reduce-zero-label) 作用见这个链接

## 注册数据集

在 `mmseg/datasets/__init__.py` 添加 `from .HomeworkDataset import HomeworkDataset` 和 `'HomeworkDataset'`

## 生成配置文件

```python
from mmengine import Config

cfg_path = './configs/pspnet/pspnet_r50-d8_1xb32-HomeworkDataset.py'

cfg = Config.fromfile(cfg_path)

cfg.norm_cfg = dict(type='BN', requires_grad=True) # 只使用GPU时，BN取代SyncBN
cfg.crop_size = (256, 256)
cfg.model.data_preprocessor.size = cfg.crop_size
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head

# 模型 decode/auxiliary 输出头，指定为类别个数
cfg.model.decode_head.num_classes = 6
cfg.model.auxiliary_head.num_classes = 6

cfg.train_dataloader.batch_size = 32

cfg.test_dataloader = cfg.val_dataloader

# 结果保存目录
cfg.work_dir = './work_dirs/pspnet'

# 训练迭代次数
cfg.train_cfg.max_iters = 9000
# 评估模型间隔
cfg.train_cfg.val_interval = 400
# 日志记录间隔
cfg.default_hooks.logger.interval = 100
# 模型权重保存间隔
cfg.default_hooks.checkpoint.interval = 1500

# 随机数种子
cfg['randomness'] = dict(seed=42)

print(cfg.pretty_text)

cfg.dump('./my_configs/pspnet_HomeworkDataset_20230616.py')


```


## 评测指标

```bash
Loads checkpoint by local backend from path: work_dirs/pspnet/iter_9000.pth
06/18 16:45:18 - mmengine - INFO - Load checkpoint from work_dirs/pspnet/iter_9000.pth
06/18 16:46:44 - mmengine - INFO - per class results:
06/18 16:46:44 - mmengine - INFO - 
+------------+-------+-------+
|   Class    |  IoU  |  Acc  |
+------------+-------+-------+
| background |  91.0 | 92.18 |
|    red     |  93.7 | 97.95 |
|   green    | 60.05 |  92.1 |
|   white    | 81.88 | 90.53 |
| seed-black | 74.24 | 85.98 |
| seed-white | 57.59 | 71.57 |
+------------+-------+-------+
06/18 16:46:44 - mmengine - INFO - Iter(test) [11/11]    aAcc: 93.5200  mIoU: 76.4100  mAcc: 88.3800  data_time: 0.2213  time: 7.7713
```



