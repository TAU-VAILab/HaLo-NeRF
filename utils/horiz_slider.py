from collections import namedtuple
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import torch
import cv2
import numpy as np
from PIL import Image

class SkyDetector:
    
    def __init__(self,
                    device='cuda'):
        self.fe = SegformerImageProcessor.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b3-finetuned-ade-512-512")
        self.model.to(device)
        self.model.eval();
    
    def has_sky(self, img):
        with torch.no_grad():
            inp = self.fe(images=img, return_tensors='pt').to('cuda')
            out = self.model(**inp)
            L = out.logits
        
        return 2 in L[0].argmax(dim=0).cpu().unique()
        # 2 is "sky" label in ade20k: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv

class HorizSlider:
    
    def __init__(self,
                    device='cuda',
                    CKPT='../correspondences/output/clipseg_ft_crops_10epochs',
                    BASE_CKPT='CIDAS/clipseg-rd64-refined',
                    stride=0.1):
        
        self.ff = FacadeFinder(device=device)
        self.processor = AutoProcessor.from_pretrained(BASE_CKPT)
        self.model = CLIPSegForImageSegmentation.from_pretrained(CKPT)
        self.model.to(device)
        self.model.eval()
        
        self.stride = stride
        
        self.sky_detector = SkyDetector(device=device)
        
    def segment(self, img, label, building_type='cathedral', debug=False):
        
        is_outdoor = self.sky_detector.has_sky(img)
        if debug:
            print('is_outdoor?', is_outdoor)
        
        if is_outdoor:
        
            ff_out = self.ff.find_facade(img, building_type=building_type, get_bbox=True)
            if ff_out is None:
                is_outdoor = False
                if debug:
                    print('no facade found, switching is_outdoor to False')
            else:
                fseg, cutout, bbox, bbox_m = ff_out
                img_c = img.crop((bbox_m.y0, bbox_m.x0, bbox_m.y1, bbox_m.x1))
        
        if not is_outdoor:
            img_c = img
            
        w, h = img_c.size
        if w <= h:
            if debug:
                print('padding...')
            seg = self._seg_pad(img_c, label)
        else:
            if debug:
                print('cropping...')
            seg = self._seg_crop(img_c, label)
        assert seg.shape == np.asarray(img_c).shape[:2], 'segmentation shape does not match image'
        
        # pad segmentation of facade crop to match original image size:
        
        if is_outdoor:
            seg_out = np.zeros_like(img.convert('L'), dtype=np.float32)
            seg_out[bbox_m.x0:bbox_m.x1, bbox_m.y0:bbox_m.y1] = seg
            return seg_out
        else:
            return seg
            
    def _seg_crop(self, img_c, label):
        w, h = img_c.size
        STRIDE = int(h * self.stride)
        indices = [(i, 0, h+i, h) for i in range(0, w-h, STRIDE)]
        crops = [img_c.crop(x) for x in indices]

        with torch.no_grad():
            inp = self.processor(images=crops, text=[label] * len(crops), return_tensors="pt", padding=True).to('cuda')
            out = self.model(**inp).logits
            S_all = out.sigmoid().cpu().numpy()
            if len(S_all.shape) == 2:
                S_all = S_all[None] # edge case - only one crop => output has one less dimension

        counts = np.zeros_like(img_c.convert('L'), dtype=np.int64)
        seg = np.zeros_like(img_c.convert('L'), dtype=np.float32)
        for S_, coords in zip(S_all, indices):
            S_ = cv2.resize(S_, (h, h))
            x0, y0, x1, y1 = coords
            for x in range(x0, x1):
                for y in range(y0, y1):
                    dx = x - x0
                    dy = y - y0
                    counts[y, x] += 1
                    seg[y, x] += S_[dy, dx]
        seg /= np.where(counts == 0., 1., counts)
        
        return seg
        
    def _seg_pad(self, img_c, label):
        w, h = img_c.size
        delta = h - w
        pad_left = delta // 2
        pad_right = delta - pad_left

        img_p = Image.fromarray(cv2.copyMakeBorder(np.asarray(img_c), 0, 0, pad_left, pad_right, cv2.BORDER_REPLICATE))

        with torch.no_grad():
            inp = self.processor(images=[img_p], text=[label], return_tensors="pt", padding=True).to('cuda')
            out = self.model(**inp).logits
            S_p = out.sigmoid().cpu().numpy()

        seg_p = cv2.resize(S_p, img_p.size)
        seg = seg_p[:, pad_left:-pad_right] if pad_right > 0 else seg_p[:, pad_left:]
        
        return seg

BBox = namedtuple("BBox", "x0 y0 x1 y1")   
# TODO: x and y are backwards?
    
class FacadeFinder:
    
    def __init__(self,
                device='cuda',
                clipseg_proc_name='CIDAS/clipseg-rd64-refined',
                clipseg_checkpoint='CIDAS/clipseg-rd64-refined',
                clipseg_threshold=0.5
                ):
        
        self.device = device
        
        self.cs_proc = AutoProcessor.from_pretrained(clipseg_proc_name)
        self.cs = CLIPSegForImageSegmentation.from_pretrained(clipseg_checkpoint)

        self.cs.to(device)
        self.cs.eval()
        self.clipseg_threshold = clipseg_threshold
        
        
    def find_facade(self, img, building_type='cathedral', pbar=False, get_bbox=False, bbox_margin=0.1):
        
        I = np.asarray(img)
        W = np.ones_like(I) * 255 # white
        def make_masked_(S):
            S_ = S[..., None]
            IS = I * S_ + W * (1 - S_)
            return IS
        
        with torch.no_grad():
            inp = self.cs_proc(
                text=[building_type],
                images=[img],
                padding="max_length", return_tensors="pt").to('cuda')
            out = self.cs(**inp)
            S = out.logits.sigmoid().cpu().numpy()
            S = cv2.resize(S, img.size)
            fseg = S > self.clipseg_threshold
            cutout = make_masked_(fseg).astype(np.uint8)

        if not get_bbox:    
            return fseg, cutout
        
        x, y = np.where(fseg)
        
        if len(x) == 0 or len(y) == 0:
            return None
        
        x0, x1 = x.min(), x.max()
        y0, y1 = y.min(), y.max()
        
        bbox = BBox(x0, y0, x1, y1)
        
        # margins
        w = x1 - x0
        h = y1 - y0
        mx = int(w * bbox_margin)
        my = int(h * bbox_margin)
        
        mx0 = max(0, bbox.x0 - mx)
        my0 = max(0, bbox.y0 - my)
        mx1 = min(img.size[1], bbox.x1 + mx)
        my1 = min(img.size[0], bbox.y1 + my)
        
        bbox_m = BBox(mx0, my0, mx1, my1)
        
        return fseg, cutout, bbox, bbox_m