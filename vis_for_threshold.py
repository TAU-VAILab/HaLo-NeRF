import os
from matplotlib import pyplot as plt
import torch
import pickle
from PIL import Image
import base64


import pandas as pd
dataFrame = pd.read_csv('/home/cc/students/csguests/chendudai/Thesis/data/st_paul_geometric_clip.csv')
dataFrame = dataFrame.sort_values(by="a picture of a cathedral's facade", ascending=False)
list_name = dataFrame[:150]
list_name = list_name['filename'][list_name.index]
list_names = list_name.values.tolist()

from clipseg.models.clipseg import CLIPDensePredT
from PIL import Image
from matplotlib import pyplot as plt
import torchvision.transforms as transforms



threshold = 0.25
cat = 'portals/'
prompt = ["a picture of a cathedral's " + cat.split('/')[0]]
path = '/home/cc/students/csguests/chendudai/Thesis/data/morris_npy/st_paul/'
save_dir = path + f'thresh_{threshold}/' + cat
# save_dir = path + f'vis_thresh_sorted_noThreshold/' + cat
path_cat = path + cat
path_images = path + 'images'

dir_list_cat = os.listdir(path_cat)
dir_list_images = os.listdir(path_images)

os.makedirs(save_dir, exist_ok=True)
colormap = plt.get_cmap('jet')


j = 0
semantics_gt_all = []

for i in list_names:
    print(j)
    j = j + 1

    with open(os.path.join(path_images, i), 'rb') as f:
        img = Image.open(f).convert('RGB')


    path = os.path.join(path_cat, i.replace(i.split('.')[1],'pickle'))
    try:
        with open(path, 'rb') as f:
            semantics_gt = torch.Tensor(pickle.load(f))
    except:
        continue

    img_w, img_h = img.size

    semantics_gt = torch.nn.functional.interpolate(semantics_gt.unsqueeze(dim=0).unsqueeze(dim=0), size=(img_h, img_w))
    semantics_gt = semantics_gt.squeeze(dim=0).permute(1, 2, 0)
    semantics_gt = semantics_gt.reshape(-1,)
    semantics_gt_all.append(semantics_gt)


semantics_gt_all = torch.cat(semantics_gt_all, 0)
hist = semantics_gt_all.histc(min=0.1, max=1, bins=10000)
mean = semantics_gt_all.mean()
max_n = semantics_gt_all.max()
min_n = semantics_gt_all.min()
median = semantics_gt_all.median()
new_x_ticks = torch.linspace(0,1,len(hist))
plt.plot(new_x_ticks, hist)
plt.title(f'st_paul, categery: {cat[:-1]}, min: {round(float(min_n),2)}, max: {round(float(max_n),2)}, median: {round(float(median),2)}, mean: {round(float(mean), 2)}')
plt.xlim([0, 1])
plt.ylim([0, 10000])
plt.show()


