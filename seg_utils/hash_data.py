from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json
from glob import glob
from PIL import Image
import os
import cv2

def resize_img(img, max_dim):
    # resize image so no dimension is greater than max_dim
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    r = max_dim / m
    w_new, h_new = int(w * r), int(h * r)
    return img.resize((w_new, h_new))


class HashDatum:
    
    def __init__(self, name, img, L, fns, name2spl, name2bt, h=None, resize_to=(352, 352)):
        self.hash = h
        
        self.name = name
        self.spl = name2spl[self.name]
        self.building_type = name2bt[self.name]
        self.img = img
        self.label = L
        self.fns = fns
        
        self.pos_fns = [x for x in self.fns if '_neg_' not in x]
        self.neg_fns = [x for x in self.fns if '_neg_' in x]
        self.has_neg = len(self.neg_fns) > 0
        
        self.prefix = self.pos_fns[0].split('_')[0]
        
        self.mask_fn = self.prefix + f'_mask_{L}.npy'
        self.gt_fn = self.prefix + f'_{L}.npy'
        self.metadata_fn = self.prefix + f'_{L}.json'
        
        self.gt = np.load(self.gt_fn)
        self.mask = np.load(self.mask_fn)
        with open(self.metadata_fn, 'r') as f:
            self.metadata = json.load(f)
        
        self.gt_ = cv2.resize(self.gt, resize_to)
        self.mask_ = (cv2.resize(self.mask.astype(float), resize_to)).astype(int)
            
        if self.has_neg:
            self.neg_mask_fn = self.prefix + f'_neg_mask_{L}.npy'
            self.neg_gt_fn = self.prefix + f'_neg_{L}.npy'
            
            self.neg_gt = np.load(self.neg_gt_fn)
            self.neg_mask = np.load(self.neg_mask_fn)
            
            self.neg_gt_ = cv2.resize(self.neg_gt, resize_to)
            self.neg_mask_ = (cv2.resize(self.neg_mask.astype(float), resize_to)).astype(int)

class HashData:
    
    def __init__(self, h, name2spl, name2bt, dirname='output/dataset', **kwargs):
        self.hash = h
        fns_ = glob(f'{dirname}/*/{h}*')
        self.name = fns_[0].split('/')[2]
        self.spl = name2spl[self.name]
        img_fn = [x for x in fns_ if '_' not in x][0]
        self.img = Image.open(img_fn).convert('RGB')
        labels = [os.path.splitext(x.split('_')[-1])[0] for x in fns_ if '_' in x]
        self.data = {
            L: HashDatum(self.name, self.img, L, [fn for fn in fns_ if f'_{L}.' in fn], name2spl, name2bt, h=h, **kwargs)
            for L in labels
        }

class HashDS(Dataset):
    
    def __init__(self, dirname, hash_metadata_filename, name2spl, name2bt, neg_only=False, spl='train'):
        df = pd.read_csv(hash_metadata_filename)
        if neg_only:
            df = df[df.has_neg]
        self.df = df[df.spl == spl].reset_index(drop=True)
        self.dirname = dirname
        self.name2spl = name2spl
        self.name2bt = name2bt
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        h = row.hash
        label = row.label
        hd = HashData(h, self.name2spl, self.name2bt, dirname=self.dirname)
        x = hd.data[row.label_raw]
        x.img_ = resize_img(x.img, 500)
        x.raw_label = x.label
        x.label = row.label # overwrite it from being the raw label
        return x