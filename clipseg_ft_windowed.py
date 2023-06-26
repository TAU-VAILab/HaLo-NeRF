from slider import Slider, resize_img
import cv2
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
import pandas

def get_k_files(k, csv_path, prompt, scene2save):
    xls_file = pandas.read_csv(csv_path)
    xls_file = xls_file[xls_file["building"]  == scene2save]
    col = xls_file[prompt]
    col_sorted_descending = col.sort_values(ascending=False)
    files_pos = col_sorted_descending[:k]
    names_pos = xls_file['base_fn'][files_pos.index]
    return names_pos.values.tolist()



scene2save = 'st_paul/windowed'
cat = ["portals","windows","towers"]
path_images = '/home/cc/students/csguests/chendudai/Thesis/data/st_paul/dense/images/'
csv_path = '/home/cc/students/csguests/chendudai/Thesis/data/st_paul_geometric_occlusions.csv'
BT = 'cathedral' #'mosque'
save_images = False

# CKPT = '/storage/morrisalper/notebooks/babel/checkpoints/clipseg_ft_crops_10epochs'
CKPT = '/home/cc/students/csguests/chendudai/Thesis/data/clipseg_ft_crops_10epochs/'
# clipseg-base: CIDAS/clipseg-rd64-refined
# clipseg-ft (old): /storage/morrisalper/notebooks/babel/correspondences/output/ckpts/clipseg_ft
# clipseg-ft (new, 10 epochs): /storage/morrisalper/notebooks/babel/checkpoints/clipseg_ft_crops_10epochs
res = 150
colormap = plt.get_cmap('jet')
k = 50
imgs_list = get_k_files(k, csv_path, "a picture of a cathedral's facade")

for c in cat:
    print(c)
    label = c

    folder2save = os.path.join(os.path.join(CKPT, scene2save), c)
    os.makedirs(folder2save, exist_ok=True)


    i = 0
    for img_name in imgs_list:
        print(i)
        i = i + 1
        img = Image.open(os.path.join(path_images,img_name)).convert('RGB')
        img = resize_img(img, 500)
        s = Slider(clipseg_checkpoint=CKPT)

        try:
            s.cache_facade(img, building_type=BT)
            sliderres = s.process(img, label, wh=(res, res))
            Zf = s.res2seg(img, sliderres, wh=(res, res))
        except AssertionError:
            Zf = s.process_unwindowed(img, label)

        name = img_name.split('.')[0]
        mask = cv2.resize(Zf, (352, 352))
        with open(os.path.join(folder2save, name + '.pickle'), 'wb') as handle:
            torch.save(mask, handle)
        if save_images:
            plt.imsave(os.path.join(folder2save, name + '_pred_clipseg.png'), mask, cmap=colormap)
            img.save(os.path.join(folder2save, name + '.png'))






