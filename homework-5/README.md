# 准备环境

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip3 install openmim
mim install 'mmcv>=2.0.0'
mim install 'mmengine'
mim install 'mmagic'
git clone https://github.com/open-mmlab/mmagic.git
cd mmagic
pip3 install -e .
pip install opencv-python pillow matplotlib seaborn tqdm
pip install clip transformers gradio 'httpx[socks]' diffusers==0.14.0 accelerate
mim install 'mmdet>=3.0.0'
```

# 准备文件夹和图片

```bash
mkdir data/
```

# 准备脚本和 Prompt

```python
import os
import cv2
import numpy as np
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules

register_all_modules()

cfg = Config.fromfile('configs/controlnet/controlnet-canny.py')

controlnet = MODELS.build(cfg.model).cuda()

control_url = 'data/maopifang.jpg'
control_img = mmcv.imread(control_url)
control = cv2.Canny(control_img, 100, 200)
control = control[:, :, None]
control = np.concatenate([control] * 3, axis=2)
control = Image.fromarray(control)

prompt = 'Warm walls, chandeliers, and brown doors.'

output_dict = controlnet.infer(prompt, control=control)
samples = output_dict['samples']
for idx, sample in enumerate(samples):
    sample.save(f'data/sample_{idx}.png')
controls = output_dict['controls']
for idx, control in enumerate(controls):
    control.save(f'data/ontrol_{idx}.png')
```

# 结果

[data/](data/)