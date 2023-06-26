import argparse
# from datasets import WikiScenesDataset
from datasets.phototourism_mask_grid_sample import PhototourismDataset
from wikiscenes_utils import create_nerf_root_dir_from_ws
import os
import pickle
import shutil
from datasets.colmap_utils import read_model, write_model
import csv
import numpy as np
from PIL import Image



os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='input directory of datatset (WikiScenes3D)')
    # parser.add_argument('--root_dir', type=str, required=True,
    #                     help='root directory of dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()

    # root_dir = args.root_dir
    input_dir = args.input_dir

    # os.makedirs(root_dir, exist_ok=True)
    # sparse_folder = os.path.join(root_dir, 'dense', 'sparse')
    # os.makedirs(sparse_folder, exist_ok=True)
    cameras, images, points3D = read_model(input_dir, '.txt')
    # images_folder = os.path.join(root_dir, 'dense', 'images')
    # os.makedirs(images_folder, exist_ok=True)
    images_input_dir = os.path.join(input_dir, '../../../..', 'WikiScenes1200px', 'cathedrals')
    image_file_list = []
    real_image_name_list = []

    j = 0
    for i, image in enumerate(images):
        image_curr_input = os.path.join(images_input_dir, images[image].name)
        image_output_filename = str(i).zfill(4) + os.path.splitext(images[image].name)[1]  # + '_' + os.path.basename(images[image].name)
        # images[image] = images[image]._replace(name=image_output_filename)
        # images[image] = images[image]._replace(id=i)
        if os.path.exists(image_curr_input):
            # shutil.copyfile(image_curr_input, os.path.join(images_folder, image_output_filename))
            image_file_list.append(image_output_filename)
            real_image_name_list.append(image_curr_input.split('cathedrals')[1][1:])
        else:
            print('error!!! This file does not exist: ' + image_curr_input)
            continue


    with open("names_98_3.txt", "w", encoding="utf-8") as f:
        for i in range(len(image_file_list)):
            f.write(image_file_list[i])
            f.write('\t')
            f.write(real_image_name_list[i])
            f.write('\n')






