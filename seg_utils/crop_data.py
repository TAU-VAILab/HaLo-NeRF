from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
import pandas as pd
from glob import glob
from PIL import Image
import cv2
import os
from collections import namedtuple
import torch

CropTuple = namedtuple('CropTuple', 'C x0 y0 x1 y1 x0_ y0_ x1_ y1_')

def rand_crop(img_, h=352, w=352):
    x, y = img_.size
    crop_size = 150 + np.random.randint(-20, 20)
    x_, y_ = np.random.randint(x), np.random.randint(y)
    x0 = max(0, x_ - (crop_size // 2))
    y0 = max(0, y_ - (crop_size // 2))
    x1 = min(x, x_ + (crop_size // 2))
    y1 = min(y, y_ + (crop_size // 2))
    C = img_.crop((x0, y0, x1, y1))
    return CropTuple(C,
                     int(w * x0 / x), int(w * y0 / y), int(h * x1 / x), int(h * y1 / y),
                     x0 / x, y0 / y, x1 / x, y1 / y
    )

def search_crops(img_, label, clip_proc, clip, n_crops=10):
    crops = [rand_crop(img_) for _ in range(n_crops)]
    with torch.no_grad():
        inp = clip_proc(
            images=[x.C for x in crops],
            text=[label], return_tensors='pt',
            padding=True
        ).to('cuda')
        out = clip(**inp)
        logits = out.logits_per_image / clip.logit_scale.exp()
        logits = logits.cpu().numpy()[:, 0]
        
    return crops[logits.argmax().item()]
