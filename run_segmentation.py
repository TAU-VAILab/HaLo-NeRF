from utils.horiz_slider import *
import cv2
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
import pandas
import base64
from tqdm import tqdm
from config.seg_config import get_opts

def print_img(image_path, output_file):
    """
    Encodes an image into html.
    image_path (str): Path to image file
    output_file (file): Output html page
    """
    if os.path.exists(image_path):
        img = base64.b64encode(open(image_path, "rb").read()).decode("utf-8")
        print(
            '<img src="data:image/png;base64,{0}">'.format(img),
            file=output_file,
        )


def get_k_files(k, csv_path):
    prompt = ["score"]
    csv_file = pandas.read_csv(csv_path)
    col = csv_file[prompt]
    col_sorted_descending = col.sort_values(by=prompt, ascending=False)
    files_pos = col_sorted_descending[:k]
    names_pos = csv_file['fn'][files_pos.index]
    return names_pos.values.tolist()


args = get_opts()
cat = args.prompts.split(';')
BT = args.building_type
folder_to_save = args.folder_to_save
images_folder = args.images_folder
use_csv_for_retrieval = args.use_csv_for_retrieval
csv_path = args.csv_retrieval_path
k = args.n_files
save_images = args.save_images
save_baseline = args.save_baseline
save_refined_clipseg = args.save_refined_clipseg
model_path = args.model_path


CKPT = model_path
CKPT_BASE = 'CIDAS/clipseg-rd64-refined'  #data_folder + 'clipseg-base_model/
colormap = plt.get_cmap('jet')
hs = HorizSlider(CKPT=CKPT)
hs_base = HorizSlider(CKPT=CKPT_BASE)


for c in cat:
    print(c)
    label = c
    if use_csv_for_retrieval:
        if not os.path.exists(csv_path):
            print('csv does not exist!')
            continue
        imgs_list = get_k_files(k, csv_path)
    else:
        try:
            imgs_list = os.listdir(images_folder)
        except:
            print(f"indoor label {label} is not in the csv!")
            continue

    folder2save_clipseg_base = os.path.join(os.path.join(folder_to_save, 'clipseg_base'), c)
    folder2save_clipseg_ft = os.path.join(os.path.join(folder_to_save, 'clipseg_ft'), c)
    os.makedirs(folder2save_clipseg_base, exist_ok=True)
    os.makedirs(folder2save_clipseg_ft, exist_ok=True)

    if save_images:
        # save HTML
        html_out = open(os.path.join(folder2save_clipseg_ft, "clipseg_ft_horiz.html"), "w")
        print('<head><meta charset="UTF-8"></head>', file=html_out)
        print("<h1>Results</h1>", file=html_out)

    for i in tqdm(range(len(imgs_list))):
        img_name = imgs_list[i]
        img = Image.open(os.path.join(images_folder,img_name)).convert('RGB')
        try:
            seg = hs.segment(img, label, building_type=BT)
            if save_baseline:
                seg_base = hs_base.segment(img, label, building_type=BT)

        except:
            print("error!")
            continue

        name = img_name.split('.')[0]
        img = img.resize((img.size[0] // 2, img.size[1] // 2))

        seg = cv2.resize(seg, (img.size[0], img.size[1]))
        if save_baseline:
            seg_base = cv2.resize(seg_base, (img.size[0], img.size[1]))

        if save_refined_clipseg:
            with open(os.path.join(folder2save_clipseg_ft, name + '.pickle'), 'wb') as handle:
                torch.save(seg, handle)
        if save_baseline:
            with open(os.path.join(folder2save_clipseg_base, name + '.pickle'), 'wb') as handle:
                torch.save(seg_base, handle)
        if save_images:
            fig = plt.figure()
            fig, axis = plt.subplots(1,5, figsize=(20,4))
            fig.suptitle(f'category: {c}, retreival order: {i}')
            axis[0].imshow(img)
            axis[0].title.set_text('rgb gt')

            im = axis[1].imshow(seg, cmap=colormap)
            axis[1].title.set_text(f'clipseg ft pred')
            axis[2].imshow(img)

            seg_thresh = seg
            seg_thresh[seg_thresh < 0.2] = 0
            seg_thresh[seg_thresh >= 0.2] = 1

            axis[2].imshow(seg_thresh, cmap=colormap, alpha=0.5)
            axis[2].title.set_text(f'clipseg ft pred overlay')

            if save_baseline:
                axis[3].imshow(seg_base, cmap=colormap)
                axis[3].title.set_text(f'clipseg base pred')
                axis[4].imshow(img)
                axis[4].imshow(seg_base, cmap=colormap, alpha=0.5)
                axis[4].title.set_text(f'clipseg base pred overlay')

            for ax in axis:
                ax.axis('off')
            plt.tight_layout()
            fig.colorbar(im)

            path2save = os.path.join(folder2save_clipseg_ft, name + '_pred_clipseg.png')
            plt.savefig(path2save)
            print_img(path2save, html_out)
            print(f"<br><b>{os.path.basename(path2save)}</b><br>", file=html_out)
            os.remove(os.path.join(folder2save_clipseg_ft, name + '_pred_clipseg.png'))

    if save_images:
        print("<hr>", file=html_out)
        html_out.close()







