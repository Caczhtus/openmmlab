# 步骤

https://bbs.csdn.net/topics/615737655

# mmdet 指标

```bash
DONE (t=0.02s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.537
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.954
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.501
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.537
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.598
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.650
06/04 21:46:45 - mmengine - INFO - bbox_mAP_copypaste: 0.537 0.954 0.501 -1.000 -1.000 0.537
06/04 21:46:45 - mmengine - INFO - Epoch(val) [50][2/2]    coco/bbox_mAP: 0.5370  coco/bbox_mAP_50: 0.9540  coco/bbox_mAP_75: 0.5010  coco/bbox_mAP_s: -1.0000  coco/bbox_mAP_m: -1.0000  coco/bbox_mAP_l: 0.5370  data_time: 2.0657  time: 2.1472
```

### mmpose 指标

```bash
DONE (t=0.00s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.741
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.970
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.741
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.793
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.976
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.793
06/05 01:37:52 - mmengine - INFO - Evaluating PCKAccuracy (normalized by ``"bbox_size"``)...
06/05 01:37:52 - mmengine - INFO - Evaluating AUC...
06/05 01:37:52 - mmengine - INFO - Evaluating NME...
06/05 01:37:52 - mmengine - INFO - Epoch(val) [200][6/6]    coco/AP: 0.740731  coco/AP .5: 1.000000  coco/AP .75: 0.970297  coco/AP (M): -1.000000  coco/AP (L): 0.740731  coco/AR: 0.792857  coco/AR .5: 1.000000  coco/AR .75: 0.976190  coco/AR (M): -1.000000  coco/AR (L): 0.792857  PCK: 0.975057  AUC: 0.141893  NME: 0.039382  data_time: 0.408166  time: 0.432510
06/05 01:37:52 - mmengine - INFO - The previous best checkpoint /data/run01/scz0brk/openmmlab/mmpose/work_dirs/rtmpose-s-ear/best_PCK_epoch_170.pth is removed
```