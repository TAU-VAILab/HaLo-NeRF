
from transformers import AutoProcessor, CLIPSegForImageSegmentation
import torch
import cv2
import numpy as np
from collections import Counter

def get_cc(seg_th):
    _, cc = cv2.connectedComponents(seg_th.numpy().astype(np.uint8))
    counts = Counter(x for x in cc.ravel() if x != 0)
    items = counts.items()
    if len(items) == 0:
        return cc == -1
    k, _ = sorted(counts.items(), key=lambda x: x[1])[-1]
    return cc == k

def cc2margins(cc):
    w, h = cc.shape
    margins = []
    for c in [cc, cc[::-1, ::-1]]:
        for i in range(w):
            x = np.where(c[i])[0]
            if len(x) > 0:
                margins.append(x[0] / h)
            else:
                margins.append(1.0)
        for i in range(h):
            x = np.where(c[:, i])[0]
            if len(x) > 0:
                margins.append(x[0] / w)
            else:
                margins.append(1.0)
    return np.array(margins)

class GeometricRetriver:
    
    def __init__(self, CKPT='CIDAS/clipseg-rd64-refined', THRESH=0.5, prompt='cathedral'):
        self.proc = AutoProcessor.from_pretrained(CKPT)
        self.model = CLIPSegForImageSegmentation.from_pretrained(CKPT)
        self.model.to('cuda');
        self.model.eval();
        self.THRESH = THRESH
        self.prompt = prompt
    
    def process(self, img):
        with torch.no_grad():
            inp = self.proc(images=img, text=self.prompt, return_tensors="pt").to('cuda')
            out = self.model(**inp)
            seg = out.logits.sigmoid().cpu()
        
        seg_th = seg > self.THRESH
        cc = get_cc(seg_th)
        margins = cc2margins(cc)
        
        scores = (cc.mean(), margins.min(), np.median(margins))
        
        return scores