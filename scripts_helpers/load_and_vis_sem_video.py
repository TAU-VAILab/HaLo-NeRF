import imageio
import numpy as np
from matplotlib.colors import ListedColormap
import os
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import torch
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

def unique(list1):
    unique_list = []
    for x in list1:
        try:
            y = int(x)
        except:
            continue
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

# path = '/storage/chendudai/repos/HaLo-NeRF/save/results/phototourism/blue_mosque_minarets_changeApp_72frames/'
path = '/storage/chendudai/repos/HaLo-NeRF/save/results/phototourism/st_paul_pediment_041023_baseline_try2'
path2save = '/storage/chendudai/repos/HaLo-NeRF/save/results/phototourism/st_paul_pediment_041023_baseline_try2_vid'
path_app = '/storage/chendudai/repos/HaLo-NeRF/save/results/phototourism/st_paul_appearance_041023_pediment_try2/'
save_appearance_change = False
freeze_appearance = True
use_sky_mask = False
blur_pred = True

proc = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b1-finetuned-ade-512-512")
os.makedirs(path2save, exist_ok=True)
l_dir = os.listdir(path)
l_dir = [f[:3]for f in l_dir]
l_dir = unique(l_dir)
l_dir = sorted(l_dir)

results_green = []
results_yellow = []
results_green_gray = []
results_yellow_gray = []
imgs = []


C = np.zeros((256, 4), dtype=np.float32)
C[:, 1] = 1.
C[:, -1] = np.linspace(0, 1, 256)
cmap_ = ListedColormap(C)
x = 100 / 1.3


# l_dir = l_dir[28:] # blue_mosque minartes
for num in l_dir:
    print(num)
    path_pred = os.path.join(path,f'{num}_semantic.png')
    path_img = os.path.join(path,f'{num}.png')

    pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)

    if blur_pred:
        w, h = pred.shape
        m = min(w, h)
        r = 500 / m
        pred = cv2.resize(pred, (int(h * r), int(w * r)))
        pred = cv2.GaussianBlur(pred, (11, 11), 10)
        # pred = cv2.GaussianBlur(pred, (21, 21), 20)
        pred = cv2.resize(pred, (h, w))

    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if use_sky_mask:
        with torch.no_grad():
            inp = proc(images=img, return_tensors='pt')
            out = model(**inp)
        L = out.logits[0]
        S = L.softmax(dim=0)[2].numpy()
        S = cv2.resize(S, (pred.shape[1], pred.shape[0]))
        pred = pred * (1 - S)

    w, h, _ = img.shape
    figsize = h / float(x), w / float(x)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.imshow(pred, cmap=cmap_, alpha=np.asarray(pred) / 255)
    plt.axis('off')
    plt.savefig(os.path.join(path2save,f'{num}.png'), bbox_inches="tight", pad_inches=0)
    plt.close()

    result_green = Image.open(os.path.join(path2save,f'{num}.png')).convert('RGB')
    result_green = result_green.resize((h, w))
    results_green += [result_green]
    imgs += [img]

res_iterp = []

for a in np.arange(0,1.01,0.05):
    iterp = a * img + (1-a) * result_green
    iterp = iterp.astype('uint8')
    res_iterp.append(iterp)

res_iterp.extend([img]*15)
results_green.extend(res_iterp)


# Apprearance
if save_appearance_change:
    l_dir = os.listdir(path_app)
    l_dir = [f[:3]for f in l_dir]
    l_dir = unique(l_dir)
    l_dir = sorted(l_dir)

    ## Milano windows
    # x = l_dir[::10]
    # x = x[:8]
    # l_dir = x + l_dir[75:]

    imgs = []
    for num in l_dir:
        # if int(num) % 2 == 0:   # portals notre
        #     continue
        path_pred = os.path.join(path_app,f'{num}_semantic.png')
        path_img = os.path.join(path_app,f'{num}.png')
        # pred = cv2.imread(path_pred, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(path_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        imgs += [img]
    results_green.extend(imgs)
elif freeze_appearance:
    freeze_img = []
    l_dir = os.listdir(path_app)
    l_dir = [f[:3]for f in l_dir]
    l_dir = unique(l_dir)
    l_dir = sorted(l_dir)

    ## Milano windows
    # x = l_dir[::10]
    # x = x[:8]
    # l_dir = x + l_dir[75:]

    freeze_img.extend([img] * (len(l_dir))) # // 2)
    results_green.extend(freeze_img)




results_green_rev = results_green[::-1]
results_green_with_rev = results_green + results_green_rev

imageio.mimsave(os.path.join(path2save, 'video.mp4'), results_green_with_rev, fps=24)
