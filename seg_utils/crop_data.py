from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
from PIL import Image
from collections import namedtuple
import torch
import os

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


CropRes = namedtuple("CropRes", "img, coords, coords_, label")

class CropDS(IterableDataset):
    
    def __init__(self, data_dir, crop_metadata_filename, res=352):
        self.data_dir = data_dir
        self.df = pd.read_csv(crop_metadata_filename)
        self.res = res
        
    def __iter__(self):
        while True:
            row = self.df.sample().iloc[0]
            fn = os.path.join(self.data_dir, row.fn)
            img = Image.open(fn).convert('RGB')
            coords = eval(row.coords)
            label = row.label
            w, h = img.size
            coords_ = (coords[0] / w, coords[1] / h, coords[2] / w, coords[3] / h)
            coords_ = tuple(int(x * self.res) for x in coords_)
            yield CropRes(img, coords, coords_, label)