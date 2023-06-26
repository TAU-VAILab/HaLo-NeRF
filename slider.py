from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, CLIPSegForImageSegmentation
# from segment_anything import build_sam, SamAutomaticMaskGenerator
import torch
import numpy as np
from tqdm.auto import tqdm
from collections import namedtuple
import cv2

def resize_img(img, max_dim=500):
    w, h = img.size
    m = max(w, h)
    if m <= max_dim:
        return img
    r = max_dim / m
    w_new, h_new = int(w * r), int(h * r)
    return img.resize((w_new, h_new))

def get_patches(img, w, h, sw, sh):
    xs = np.arange(0, img.size[0], sw)
    ys = np.arange(0, img.size[1], sh)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if x + w < img.size[0] and y + h < img.size[1]:
                coords = (x, y, x + w, y + h)
                ci = img.crop(coords)
                yield i, j, ci, coords

SliderRes = namedtuple("SliderRes", "patches scores scoremap patchsegs coords")
                
class Slider:
    
    def __init__(self,
                 clip_proc_name='openai/clip-vit-base-patch32',
                 # clip_checkpoint='../ckpts/monolingual-uncased-unk-filt_bsz128_lr1e-06_5epochs/last/0_CLIPModel',\
                 clip_checkpoint='/home/cc/students/csguests/chendudai/Thesis/data/clipseg-ft_model/0_CLIPModel/',
                 clipseg_proc_name='CIDAS/clipseg-rd64-refined',
                 clipseg_checkpoint='../ckpts/clipseg_ft',
                 device='cuda',
                 # device='cpu',
                 **kwargs
                ):
        self.clip_proc = CLIPProcessor.from_pretrained(clip_proc_name)
        self.clip = CLIPModel.from_pretrained(clip_checkpoint)
        self.cs_proc = AutoProcessor.from_pretrained(clipseg_proc_name)
        self.cs = CLIPSegForImageSegmentation.from_pretrained(clipseg_checkpoint)
        
        self.clip.to(device)
        self.clip.eval()
        
        self.cs.to(device)
        self.cs.eval()
        
        self._ls = self.clip.logit_scale.exp().item()
        
        self.device = device
        
        self.ff = FacadeFinder(device=device, **kwargs)
        
        self.fseg, self.cutout = None, None
    
    def cache_facade(self, img, building_type='cathedral'):
        self.fseg, self.cutout = self.ff.find_facade(img, building_type=building_type)
    
    def process_unwindowed(self, img, prompt, fseg_mult=True):
        assert self.fseg is not None, 'fseg needed'
        with torch.no_grad():
            inp = self.cs_proc(
                text=[prompt],
                images=img,
                padding="max_length", return_tensors="pt").to('cuda')
            out = self.cs(**inp)
            S = out.logits.sigmoid().cpu().numpy()
        S_resized = cv2.resize(S, (self.fseg.shape[1], self.fseg.shape[0]))
        return S_resized * self.fseg if fseg_mult else S_resized
    
    def process(self, img, prompt, bsz=64, wh=(100, 100), strides=(25, 25), pbar=False):
        # img: PIL
        w, h = wh
        sw, sh = strides
        data = list(get_patches(img, w, h, sw, sh))
        
        assert len(data) > 0
        
        I, J, patches, coords = zip(*data)
        
        batches = []
        for i in range(0, len(patches), bsz):
            batches.append(patches[i:i+bsz])
        
        Ls = []
        with torch.no_grad():
            for B in (tqdm(batches) if pbar else batches):
                inp = self.clip_proc(
                    images=B, text=prompt, return_tensors="pt").to(self.device)
                out = self.clip(**inp)
                Ls.append(out.logits_per_image[:, 0].cpu().numpy() / self._ls)
        
        L = np.hstack(Ls)
        
        X = np.zeros((max(I) + 1, max(J) + 1))
        for k, (i, j) in enumerate(zip(I, J)):
            X[i, j] = L[k]
        X = X.T
        
        Ss = []
        with torch.no_grad():
            for B in (tqdm(batches) if pbar else batches):
                inp = self.cs_proc(
                    text=[prompt] * len(B),
                    images=B,
                    padding="max_length", return_tensors="pt").to('cuda')
                out = self.cs(**inp)
                Ss.append(out.logits.sigmoid().cpu().numpy())
        S = np.vstack(Ss)
        
        return SliderRes(patches, L, X, S, coords)
    
    def res2seg(self, img, sliderres, wh=(100, 100), strides=(25, 25), score_threshold=None, pbar=False, use_fseg=True):
        patches, L, X, S, coords = sliderres
        
        T = np.mean(L) if score_threshold is None else score_threshold
        # default score threshold: mean
        
        w, h = wh
        sw, sh = strides
        
        Z = np.zeros((img.size[1], img.size[0]))
        counts = Z.copy()
        for i, p in enumerate((tqdm(patches) if pbar else patches)):
            if L[i] > T:
                seg = S[i]
                seg_ = cv2.resize(seg, (w, h))
                # print(i, L[i], coords[i], seg.shape)
                x0, y0, x1, y1 = coords[i]
                for j, x in enumerate(range(x0, x1)):
                    for k, y in enumerate(range(y0, y1)):
                        if y < counts.shape[0] and x < counts.shape[1]:
                            counts[y, x] += 1

                            Z[y, x] += seg_[k, j]
        
        Z_ = Z / np.where(counts == 0, 1, counts)
        
        if use_fseg:
            assert self.fseg is not None, 'missing facade segmentation'
            Z_ *= self.fseg
        
        return Z_
    

class FacadeFinder:
    
    def __init__(self,
                clip_checkpoint='openai/clip-vit-base-patch32',
                sam_checkpoint='../SAM/checkpoints/sam_vit_h_4b8939.pth',
                device='cpu',
                clipseg_proc_name='CIDAS/clipseg-rd64-refined',
                clipseg_checkpoint='CIDAS/clipseg-rd64-refined',
                clipseg_threshold=0.2
                ):
        
        self.device = device
        
        self.cs_proc = AutoProcessor.from_pretrained(clipseg_proc_name)
        self.cs = CLIPSegForImageSegmentation.from_pretrained(clipseg_checkpoint)

        self.cs.to(device)
        self.cs.eval()
        self.clipseg_threshold = clipseg_threshold
        
        
    def find_facade(self, img, building_type='cathedral', pbar=False):
        
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

            return fseg, cutout