import os
import torch
from transformers import AutoProcessor, CLIPSegForImageSegmentation
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

# badshahi_mosque
scene2save = '0209_megaDepth_mosque'
cat = ["portals","windows","domes", "mianerts"]
path_images = '/home/cc/students/csguests/chendudai/Thesis/data/0209_megaDepth_mosque/dense/images/'
save_images = False

k = 50
csv_path = '/home/cc/students/csguests/chendudai/Thesis/data/retrieval_clip_ft_all_140523.csv'

# CKPT = '/storage/morrisalper/notebooks/babel/checkpoints/clipseg_ft_crops_10epochs'
CKPT =  '/home/cc/students/csguests/chendudai/Thesis/data/clipseg_ft_crops_10epochs/'
# clipseg-base: CIDAS/clipseg-rd64-refined
# clipseg-ft (old): /storage/morrisalper/notebooks/babel/correspondences/output/ckpts/clipseg_ft
# clipseg-ft (new, 10 epochs): /storage/morrisalper/notebooks/babel/checkpoints/clipseg_ft_crops_10epochs

colormap = plt.get_cmap('jet')
processor = AutoProcessor.from_pretrained('CIDAS/clipseg-rd64-refined')
model = CLIPSegForImageSegmentation.from_pretrained(CKPT)
# model.to('cuda')
model.eval();


# imgs_list= os.listdir(path_images)


for c in cat:
    print(c)
    labels = [c]
    imgs_list = get_k_files(k, csv_path, c, scene2save)

    folder2save = os.path.join(os.path.join(CKPT, scene2save), c)
    os.makedirs(folder2save, exist_ok=True)

    i = 0
    for img_name in imgs_list:
        print(i)
        i = i + 1
        imgs = Image.open(os.path.join(path_images,img_name))
        with torch.no_grad():
            # inp = processor(images=imgs, text=labels, return_tensors="pt", padding=True).to('cuda')
            inp = processor(images=imgs, text=labels, return_tensors="pt", padding=True)
            out = model(**inp).logits
            mask = out.sigmoid().cpu().numpy()


            name = img_name.split('.')[0]
            with open(os.path.join(folder2save, name + '.pickle'), 'wb') as handle:
                torch.save(mask, handle)

            if save_images:
                plt.imsave(os.path.join(folder2save, name + '_pred_clipseg.png'), mask, cmap=colormap)
                imgs.save(os.path.join(folder2save, name + '.png'))


