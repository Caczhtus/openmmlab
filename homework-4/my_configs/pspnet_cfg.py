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

cfg.model.pretrained = 'checkpoint/resnet50_v1c-2cccc1ad.pth'

print(cfg.pretty_text)

cfg.dump('./my_configs/pspnet_HomeworkDataset_20230616.py')

