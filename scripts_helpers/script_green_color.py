import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from models.rendering import render_rays
from models.nerf import *
from utils.interpolate_cam_path import generate_camera_path
from utils import load_ckpt

from datasets import dataset_dict
from datasets.depth_utils import *

from models.networks import E_attr
import math
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
from config.eval_config import get_opts
import cv2
torch.backends.cudnn.benchmark = True
from matplotlib.colors import ListedColormap
import argparse

'''
This script goal is to color a specific image with the green color of the prediction image.
'''

parser = argparse.ArgumentParser()
parser.add_argument("--path2save", type=str, default='save/results/phototourism/test', help="path to input video file")
parser.add_argument("--pred", type=str, default='save/results/phototourism/pantheon_exterior_text_semantic/001_semantic.png', help="path to reference video file")
parser.add_argument("--img", type=str, default='data/pantheon_exterior/dense/images/0001.jpg', help="path to the RGB image file")


def color_image(path2save, pred, img):
    # pred = 'data/clipseg_ft_crops_refined_plur_newcrops_10epochs/notre_dame/try_ff/clipseg_base/towers/3354.pickle'
    # pred = torch.load(pred)
    # transform = T.ToPILImage()
    # pred = transform(pred)
    # img = Image.open('data/notre_dame_front_facade/dense/images/3354.jpg').convert('RGB')

    img = Image.open(img).convert('RGB')
    pred = Image.open(pred)

    img = np.asarray(img)
    pred = pred.resize([img.shape[1], img.shape[0]])
    pred = np.asarray(pred)

    C = np.zeros((256, 4), dtype=np.float32)
    C[:, 1] = 1.
    C[:, -1] = np.linspace(0, 1, 256)
    cmap_ = ListedColormap(C)
    x = 100 / 1.3

    w, h = pred.shape
    m = min(w, h)
    r = 500 / m

    sem_pred = cv2.resize(pred, (int(h * r), int(w * r)))
    sem_pred = cv2.GaussianBlur(sem_pred, (11, 11), 10)
    sem_pred = cv2.resize(sem_pred, (h, w))
    sem_pred_ = (sem_pred / 255).squeeze()

    w, h, _ = img.shape
    figsize = h / float(x), w / float(x)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.imshow(pred, cmap=cmap_, alpha=np.asarray(sem_pred_))
    plt.axis('off')
    plt.savefig(os.path.join(path2save, f'img_with_pred.png'), bbox_inches="tight",
                pad_inches=0)
    plt.close()


if __name__ == "__main__":
    args = parser.parse_args()
    color_image(args.path2save, args.pred, args.img)