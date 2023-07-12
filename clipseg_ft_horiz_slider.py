from horiz_slider import *
import cv2
import os
import torch
from PIL import Image
from matplotlib import pyplot as plt
import pandas
import base64
import argparse


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
    prompt = ["a picture of a cathedral's facade"]
    xls_file = pandas.read_csv(csv_path)
    col = xls_file[prompt]
    col_sorted_descending = col.sort_values(by=prompt, ascending=False)
    files_pos = col_sorted_descending[:k]
    names_pos = xls_file['filename'][files_pos.index]
    return names_pos.values.tolist()


def get_k_files_clip(k, csv_path, prompt, scene_name):
    xls_file = pandas.read_csv(csv_path)
    xls_file = xls_file[xls_file["building"] == scene_name]
    col = xls_file[prompt]
    col_sorted_descending = col.sort_values(ascending=False)
    files_pos = col_sorted_descending[:k]
    names_pos = xls_file['base_fn'][files_pos.index]
    return names_pos.values.tolist()

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prompts', type=str, default='poles;colonnade')   #'/home/cc/students/csguests/chendudai/Thesis/data/ft_clip_sims_v0.3-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24-nodups.csv' #retrieval_clip_outdoor_020523.csv
    parser.add_argument('--scene_name', type=str, default='st_paul')
    parser.add_argument('--folder_to_save', type=str, default='st_paul/horizontal')
    parser.add_argument('--building_type', type=str, default='cathedral')  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--data_folder', type=str, default='/home/cc/students/csguests/chendudai/Thesis/data/')  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--is_geo_occ', type=bool, default=True)  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--n_files', type=int, default=150)  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--save_images', type=bool, default=False)  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--save_baseline', type=bool, default=False)  #'mosque' 'cathedral' 'synagogue'
    parser.add_argument('--save_refined_clipseg', type=bool, default=True)  #'mosque' 'cathedral' 'synagogue'


    return parser.parse_args()

args = get_opts()
cat = args.prompts.split(';')
BT = args.building_type
scene2sav = args.folder_to_save
scene_name = args.scene_name
data_folder = args.data_folder
is_geo_occ = args.is_geo_occ
k = args.n_files
save_images = args.save_images
save_baseline = args.save_baseline
save_pickle = args.save_refined_clipseg

# cat = ["pediment", "dome", "portals","spires" ,"windows", "towers", "entrance", "portico", "colonnade"] #["portals","windows","spires"]   ["clock tower", "kuppel", "columns", "gargoyle", "statue", "clock"]   #st_paul
# cat = ["portals","spires", "windows", "entrance", "main door", "statue", "roof", "pediment", "Tympanum", "buttress", "gargoyle", "relief", "Lombard bands", "jambs", "flanks", "apse"]  # milano
# cat = ["portals", "windows", "towers","entrance", "main door", "pediment", "rose window", "gargoyle", "entrance", "main door",  "statue", "Tympanum", "buttress",  "gargoyle", "jambs", "jamp figures", "Rosetta window", "crocket", "archivolt", "Gable", "lintel", "Trumeau", "pinnacles", "spires", "roof", "apse", "gate", "relief", "sculpture", "lancets"]
# cat = ["portals", "windows", "domes", "minarets", "main door", "arches", "iwan", "muqarnas", "mihrab", "jali", "arches", "gates", "towers", "qibla wall", "gold", "court", "revak", "iron chain"] # 0209
# cat = ["portals", "windows", "domes", "minarets", "main door", "arches", "iwan", "muqarnas", "mihrab", "jali", "arches", "gates", "towers", "qibla wall", "gold", "court", "revak", "iron chain"] # badshahi
# cat = ["portals", "windows", "arches", "statues", "lancets", "organ", "altar", "relics", "chapel", "vaults", "door", "ceiling", "pillars", "columns", "nave", "choir", "chancel", "ambulatory", "aisle", "painting","triforium", "apse", "abside", "stained glass windows"] # seville_indoor
# cat = ["dome", "chandelier", "portals", "windows", "arches", "statues", "lancets", "organ", "altar", "relics", "chapel", "vaults", "door", "ceiling", "pillars", "columns", "nave", "choir", "chancel", "ambulatory", "aisle", "painting","triforium", "apse", "abside", "stained glass windows"] # 62_0_undistorted
# cat = ["dome", "portals", "windows", "arches", "fence", "staircase"]  # hurba outdoor
# cat = ["dome", "windows", "arches", "fence", "staircase", "bimah", "paroket", "ark", "stained glass"]  # hurba_indoor
# cat = ["pillars"] #["colonnade", "columns", "pillars"]
# cat = ["portals", "domes", "windows"]
# cat = ["pillars", "clock", "colonnade", "plaque", "belfry", "pediment"]   #"towers"  "spires" "domes", "minarets",    "windows", "towers"  "portals"
# is_geo_occs = [False]

# data_folder = '/home/cc/students/csguests/chendudai/Thesis/data/' # '/home/cc/students/csguests/chendudai/Thesis/data/' #'/net/projects/ranalab/itailang/chen/data/'
# data_folder = '/net/projects/ranalab/itailang/chen/data/'

if is_geo_occ:
    csv_path = data_folder + scene_name + '_geometric_occlusions.csv'
    # scene2save = scene2sav + '/geoOccRetrieval'
    scene2save = scene2sav
else:
    csv_path = data_folder + 'retrieval_clip_ft_all_140523.csv'
    # scene2save = scene2sav + '/ClipFtRetrieval'
    scene2save = scene2sav

# path_images = '/home/cc/students/csguests/chendudai/Thesis/data/' + scene_name + '/dense/images/'
path_images = data_folder + scene_name+ '/dense/images/'

# CKPT = '/home/cc/students/csguests/chendudai/Thesis/data/clipseg_ft_crops_10epochs/'
CKPT = data_folder + 'clipseg_ft_crops_refined_plur_newcrops_10epochs/'
CKPT_BASE = 'CIDAS/clipseg-rd64-refined'  #data_folder + 'clipseg-base_model/

# CKPT = '/storage/morrisalper/notebooks/babel/checkpoints/clipseg_ft_crops_10epochs'
# clipseg-base: CIDAS/clipseg-rd64-refined
# clipseg-ft (old): /storage/morrisalper/notebooks/babel/correspondences/output/ckpts/clipseg_ft
# clipseg-ft (new, 10 epochs): /storage/morrisalper/notebooks/babel/checkpoints/clipseg_ft_crops_10epochs



colormap = plt.get_cmap('jet')
hs = HorizSlider(CKPT=CKPT)
hs_base = HorizSlider(CKPT=CKPT_BASE)


for c in cat:
    print(c)
    label = c
    if is_geo_occ:
        if not os.path.exists(csv_path):
            print('csv does not exist!')
            continue
        imgs_list = get_k_files(k, csv_path)
    else:
        try:
            # imgs_list = get_k_files_clip(k, csv_path, c, scene_name)
            imgs_list = os.listdir(path_images)
        except:
            print(f"indoor label {label} is not in the csv!")
            continue

    folder2save_clipseg_base = os.path.join(os.path.join(CKPT, scene2save, 'clipseg_base'), c)
    folder2save_clipseg_ft = os.path.join(os.path.join(CKPT, scene2save, 'clipseg_ft'), c)
    os.makedirs(folder2save_clipseg_base, exist_ok=True)
    os.makedirs(folder2save_clipseg_ft, exist_ok=True)

    if save_images:
        # save HTML
        html_out = open(os.path.join(folder2save_clipseg_ft, "clipseg_ft_horiz.html"), "w")
        print('<head><meta charset="UTF-8"></head>', file=html_out)
        print("<h1>Results</h1>", file=html_out)

    i = 0
    for img_name in imgs_list:
        print(f'{i}: {img_name}')
        i = i + 1
        img = Image.open(os.path.join(path_images,img_name)).convert('RGB')
        seg = hs.segment(img, label, building_type=BT)

        try:
            seg = hs.segment(img, label, building_type=BT)
            if save_baseline:
                seg_base = hs_base.segment(img, label, building_type=BT)

        except:
            print("error!")
            continue

        name = img_name.split('.')[0]
        img = img.resize((img.size[0] // 2, img.size[1] // 2))
        # img = img.resize()

        seg = cv2.resize(seg, (img.size[0], img.size[1]))
        if save_baseline:
            seg_base = cv2.resize(seg_base, (img.size[0], img.size[1]))

        if save_pickle:
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







