# Author: Hadar Elor

import os, shutil
from datasets.colmap_utils import read_model, write_model
import csv
import numpy as np
from PIL import Image


def create_nerf_root_dir_from_ws(input_dir, root_dir):
    os.makedirs(root_dir, exist_ok=True)
    sparse_folder = os.path.join(root_dir, 'dense', 'sparse')
    os.makedirs(sparse_folder, exist_ok=True)
    cameras, images, points3D = read_model(input_dir, '.txt')
    images_folder = os.path.join(root_dir, 'dense', 'images')
    os.makedirs(images_folder, exist_ok=True)
    images_input_dir = os.path.join(input_dir, '../../../..', 'WikiScenes1200px', 'cathedrals')
    image_file_list = []

    # with open(os.path.join(root_dir, 'res.txt'), 'rb') as f:
    #     images_selected = f.readlines()
    #
    # # Convert to the correct format
    # for j,i in enumerate(images_selected):
    #     images_selected[j] = i[0:8].decode("utf-8")

    # with open(os.path.join('names.txt'), 'wt', encoding='utf-8') as out_file:
    #     for i, image in enumerate(images):
    #         image_output_filename = str(i).zfill(4) + os.path.splitext(images[image].name)[1]  # + '_' + os.path.basename(images[image].name)
    #         tsv_writer = csv.writer(out_file, delimiter='\t')
    #         tsv_writer.writerow([image_output_filename, images[image].name])

    j = 0
    for i,image in enumerate(images):
        image_curr_input = os.path.join(images_input_dir, images[image].name)
        image_output_filename = str(i).zfill(4) + os.path.splitext(images[image].name)[1]  #  + '_' + os.path.basename(images[image].name)
        images[image] = images[image]._replace(name = image_output_filename)
        images[image] = images[image]._replace(id = i)
        # if os.path.exists(image_curr_input) and  images[image].name in images_selected:
        if os.path.exists(image_curr_input):
            # image_output_filename = str(j).zfill(4) + os.path.splitext(images[image].name)[1]  # + '_' + os.path.basename(images[image].name)
            # images[image] = images[image]._replace(name=image_output_filename)
            # images[image] = images[image]._replace(id=j)
            # j += 1
            shutil.copyfile(image_curr_input, os.path.join(images_folder, image_output_filename))
            image_file_list.append(image_output_filename)
        else:
            print('error!!! This file does not exist: ' + image_curr_input)
            continue


        # update camera parameteres
        cam = images[image].camera_id
        img = Image.open(image_curr_input)
        img_w_, img_h_ = img.size
        img_w, img_h = int(cameras[cam].width), int(cameras[cam].height)
        f_ = cameras[cam].params[0] * img_w_ / img_w
        cx_ = cameras[cam].params[1] * img_w_ / img_w
        cy_ = cameras[cam].params[2] * img_w_ / img_w
        k = cameras[cam].params[3]
        cameras[cam] = cameras[cam]._replace(width = img_w_)
        cameras[cam] = cameras[cam]._replace(height = img_h_)
        cameras[cam] = cameras[cam]._replace(params = np.array([f_, cx_, cy_, k]))
        if i % 100 == 0:
            print('writing ' + image_output_filename + '... ' + str(i) + '/' + str(len(images)))
    ## save (updated) files
    write_model(cameras, images, points3D, sparse_folder, '.bin')
    # save splits
    dataset_name = os.path.basename(os.path.dirname(root_dir))
    with open(os.path.join(root_dir, dataset_name + '.tsv'), 'wt') as out_file:
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerow(['filename', 'id', 'split', 'dataset'])
        for i, filename_curr in enumerate(image_file_list):
            if 'gif' in os.path.splitext(filename_curr)[1]:
                print('disregarding from tsv: ' + filename_curr)
                continue
            split = 'test' if i % 10 == 0 else 'train'
            tsv_writer.writerow([filename_curr, str(i), split, dataset_name])
    return

