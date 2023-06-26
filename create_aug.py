import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
import pickle
from models.rendering import render_rays
from models.nerf import *
from utils.interpolate_cam_path import generate_camera_path
from utils import load_ckpt
import metrics

from datasets import dataset_dict
from datasets.depth_utils import *

from models.networks import E_attr
from math import sqrt
import math
import json
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt

torch.backends.cudnn.benchmark = True
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T

from datasets.ray_utils import *

from math import sqrt, exp
import random
import imageio
import torchvision.transforms as transforms
from clipseg.models.clipseg import CLIPDensePredT

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--root_dir', type=str,
                        default='/home/cy/PNW/datasets/nerf_synthetic/lego',
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        choices=['blender', 'phototourism'],
                        help='which dataset to validate')
    parser.add_argument('--scene_name', type=str, default='test',
                        help='scene name, used as output folder name')
    parser.add_argument('--split', type=str, default='test_test',
                        choices=['val', 'test', 'test_train', 'test_test'])
    parser.add_argument('--img_wh', nargs="+", type=int, default=[500, 500],
                        help='resolution (img_w, img_h) of the image')
    # for phototourism
    parser.add_argument('--img_downscale', type=int, default=2,
                        help='how much to downscale the images for phototourism dataset')
    parser.add_argument('--use_cache', default=False, action="store_true",
                        help='whether to use ray cache (make sure img_downscale is the same)')

    parser.add_argument('--N_emb_xyz', type=int, default=15,
                        help='number of xyz embedding frequencies')
    parser.add_argument('--N_emb_dir', type=int, default=4,
                        help='number of direction embedding frequencies')
    parser.add_argument('--N_samples', type=int, default=256,
                        help='number of coarse samples')
    parser.add_argument('--N_importance', type=int, default=256,
                        help='number of additional fine samples')
    parser.add_argument('--use_disp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument('--N_vocab', type=int, default=1500,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=True, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')

    parser.add_argument('--chunk', type=int, default=32 * 1024 * 1,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--video_format', type=str, default='mp4',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')

    parser.add_argument('--save_dir', type=str, default="./",
                        help='pretrained checkpoint path to load')
    ## Semantic Nerf
    parser.add_argument('--enable_semantic', default=False, action="store_true",
                        help='whether to enable semantics')
    parser.add_argument('--num_semantic_classes', type=int, default=2,
                        help='The number of semantic classes')


    return parser.parse_args()




@torch.no_grad()
def batched_inference(models, embeddings,
                      rays, ts, N_samples, N_importance, use_disp,
                      chunk,
                      white_back,
                      **kwargs):
    """Do batched inference on rays using chunk."""
    B = rays.shape[0]
    results = defaultdict(list)

    for i in range(0, B, chunk):
        rendered_ray_chunks = \
            render_rays(models,
                        embeddings,
                        rays[i:i + chunk],
                        ts[i:i + chunk] if ts is not None else None,
                        N_samples,
                        use_disp,
                        0,
                        0,
                        N_importance,
                        chunk,
                        white_back,
                        test_time=True,
                        **kwargs)

        for k, v in rendered_ray_chunks.items():
            results[k] += [v.cpu()]

    for k, v in results.items():
        results[k] = torch.cat(v, 0)
    return results


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def main_create_aug(aug_dir, files, images_path, root_dir, ckpt_path, use_rgb_loss):

    N_a = 48
    N_emb_xyz = 15
    N_emb_dir = 4
    encode_a = True
    dataset_name = 'phototourism'
    img_wh = [500, 500]
    split = 'test_train'
    enable_semantic = True
    num_semantic_classes = 2
    N_importance = 256
    N_samples = 256
    use_disp = False
    chunk = 32768
    img_downscale = 2
    use_cache = False
    save_dir = './save_aug'

    # list_images = os.listdir(images_path)


    kwargs = {'root_dir': root_dir,
              'split': split}

    kwargs['img_downscale'] = img_downscale
    kwargs['use_cache'] = use_cache


    dataset = dataset_dict[dataset_name](**kwargs)


    embedding_xyz = PosEmbedding(N_emb_xyz - 1, N_emb_xyz)
    embedding_dir = PosEmbedding(N_emb_dir - 1, N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if encode_a:
        # enc_a
        enc_a = E_attr(3, N_a).cuda()
        load_ckpt(enc_a, ckpt_path, model_name='enc_a')
        kwargs = {}

    nerf_coarse = NeRF('coarse',
                       enable_semantic=enable_semantic, num_semantic_classes=num_semantic_classes,
                       in_channels_xyz=6 * N_emb_xyz + 3,
                       in_channels_dir=6 * N_emb_dir + 3,
                       is_test=True).cuda()
    nerf_fine = NeRF('fine',
                     enable_semantic=enable_semantic, num_semantic_classes=num_semantic_classes,
                     in_channels_xyz=6 * N_emb_xyz + 3,
                     in_channels_dir=6 * N_emb_dir + 3,
                     encode_appearance=encode_a,
                     in_channels_a=N_a,
                     is_test=True).cuda()

    load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    # dir_name = os.path.join(save_dir, f'results/{dataset_name}/aug/{scene_name}')
    dir_name = aug_dir
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(dir_name + '/images/', exist_ok=True)

    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    model.eval()
    model.load_state_dict(torch.load('weights/rd64-uni.pth', map_location=torch.device('cuda')), strict=False)

    k = 0
    for f in files:
        idx = dataset.files.id[f == dataset.files.filename]
        idx = int(idx)

        sample_original = dataset[idx]
        ts = sample_original['ts']
        rays = sample_original['rays']
        whole_img = sample_original['whole_img'].unsqueeze(0).cuda()
        kwargs['a_embedded_from_img'] = enc_a(whole_img)
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    N_samples, N_importance, use_disp,
                                    chunk,
                                    dataset.white_back,
                                    **kwargs)

        w, h = sample_original['img_wh']

        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        # img_pred_ = (img_pred * 255).astype(np.uint8)
        # plt.imshow(img_pred_)
        # plt.show()

        img_GT = np.clip(sample_original['rgbs'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_GT_ = (img_GT * 255).astype(np.uint8)
        # plt.imshow(img_GT_)
        # plt.show()
        rgb_loss = 0.5 * ((img_pred - img_GT) ** 2).mean()
        print(rgb_loss)

        if use_rgb_loss and rgb_loss > 0.1:
            return

        # imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)
        data_to_save = {}
        data_to_save['sample'] = sample_original
        data_to_save['a_embedded_from_img'] = kwargs['a_embedded_from_img']
        data_to_save['whole_img'] = whole_img
        data_to_save['rgb'] = img_GT_
        data_to_save['poses'] = sample_original['c2w']

        with open(os.path.join(dir_name, f'{k}_imgData.pickle'), 'wb') as handle:
            pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
            k = k + 1

        dataset.test_img_w, dataset.test_img_h = img_wh
        dataset.test_focal = dataset.test_img_w / 2 / np.tan(np.pi / 6)  # fov=60 degrees
        dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w / 2],
                                        [0, dataset.test_focal, dataset.test_img_h / 2],
                                        [0, 0, 1]])

        # select appearance embedding, hard-coded for each scene
        # interpolate between multiple cameras to generate path
        # dataset.poses_test = generate_camera_path(dataset)
        img = Image.open(os.path.join(root_dir, 'dense/images',
                                      dataset.image_paths[dataset.img_ids_train[idx]])).convert('RGB')  # 111 159 178 208 252 314
        img_downscale = 4
        img_w, img_h = img.size
        img_w = img_w // img_downscale
        img_h = img_h // img_downscale
        img = img.resize((img_w, img_h), Image.LANCZOS)
        toTensor = T.ToTensor()
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img = toTensor(img)  # (3, h, w)
        whole_img = normalize(img).unsqueeze(0).cuda()
        kwargs['a_embedded_from_img'] = enc_a(whole_img)

        dataset.test_appearance_idx = idx  # 85572957_6053497857.jpg
        # N_frames = 30 * 8
        N_frames = 10

        # dx1 = np.linspace(-0, 0, 2 * N_frames // 3)
        # dx2 = np.linspace(-0.3, 0.3, N_frames - 2 * N_frames // 3)
        # dx = np.concatenate((dx1, dx2))
        #
        # dy1 = np.linspace(-0, 0, N_frames // 3)
        # dy2 = np.linspace(-0.05, 0.05, N_frames // 3)
        # dy3 = np.linspace(-0, 0, N_frames - 2 * N_frames // 3)
        # dy = np.concatenate((dy1, dy2, dy3))
        #
        # dz1 = np.linspace(-0.1, 0.1, N_frames // 3)
        # dz2 = np.linspace(-0, 0, N_frames - N_frames // 3)
        # dz = np.concatenate((dz1, dz2))

        # dz = np.linspace(0, 0, N_frames)

        # theta_x1 = np.linspace(math.pi / 30, 0, N_frames // 2)
        # theta_x2 = np.linspace(0, math.pi / 30, N_frames - N_frames // 2)
        # theta_x = np.concatenate((theta_x1, theta_x2))
        #
        # theta_y = np.linspace(math.pi / 10, -math.pi / 10, N_frames)
        #
        # # theta_x = np.linspace(0, 0, N_frames)
        # # theta_y = np.linspace(0, 0, N_frames)
        # theta_z = np.linspace(0, 0, N_frames)

        dx = [-0.3, -0.2, 0.1, 0.1, 0.2, 0.3, 0,0,0,0]
        dy = [0, 0, 0, 0, 0, 0, -0.1, 0.1, 0, 0]
        dz = [0, 0, 0, 0, 0, 0, 0, 0, -0.2, 0.2]


        theta_x = np.linspace(0, 0, N_frames)
        theta_y = np.linspace(0, 0, N_frames)
        theta_z = np.linspace(0, 0, N_frames)

        dataset.poses_test = np.tile(dataset.poses_dict[idx], (N_frames, 1, 1))
        for i in range(N_frames):
            dataset.poses_test[i, 0, 3] += dx[i]
            dataset.poses_test[i, 1, 3] += dy[i]
            dataset.poses_test[i, 2, 3] += dz[i]
            dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i], theta_y[i], theta_z[i]]),
                                                       dataset.poses_test[i, :, :3])

        kwargs['output_transient'] = False

        colormap = plt.get_cmap('jet')

            # sample = dataset_test[i]
            # ts = sample['ts']
            # if int(ts[0]) != int(file[:4]):
            #     continue



        for i in tqdm(range(0, len(dataset.poses_test), 1)):
            sample = sample_original
            sample['c2w'] = torch.FloatTensor(dataset.poses_test[i])
            ts = sample_original['ts'] + i
            directions = get_ray_directions(dataset.test_img_w, dataset.test_img_h, dataset.test_K)
            rays_o, rays_d = get_rays(directions, sample['c2w'])
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near * torch.ones_like(rays_o[:, :1]),
                              far * torch.ones_like(rays_o[:, :1])],
                             1)
            sample['rays'] = rays
            sample['img_wh'] = torch.LongTensor([dataset.test_img_w, dataset.test_img_h])

            # if int(ts[0]) != int(file[:4]):
            #     continue

            # rays = sample['rays']
            # if (split == 'test_train' or split == 'test_test') and encode_a:
            #     whole_img = sample['whole_img'].unsqueeze(0).cuda()
            #     kwargs['a_embedded_from_img'] = enc_a(whole_img)
            results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                        N_samples, N_importance, use_disp,
                                        chunk,
                                        dataset.white_back,
                                        **kwargs)

            w, h = sample_original['img_wh']

            img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
            img_pred_ = (img_pred * 255).astype(np.uint8)
            # plt.imshow(img_pred_)
            # plt.show()

            imageio.imwrite(os.path.join(dir_name, 'images', f'{k}.png'), img_pred_)

            data_to_save = {}
            data_to_save['sample'] = sample
            data_to_save['a_embedded_from_img'] = kwargs['a_embedded_from_img']
            data_to_save['whole_img'] = whole_img
            data_to_save['rgb'] = img_pred_
            data_to_save['poses'] = dataset.poses_test[i, :]

            with open(os.path.join(dir_name, f'{k}_imgData.pickle'), 'wb') as handle:
                pickle.dump(data_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)
                k = k + 1



if __name__ == "__main__":
    aug_dir = './save_aug'
    files = ['0001.jpg']
    images_path = ''
    root_dir = '/home/cc/students/csguests/chendudai/Thesis/data/st_paul/'
    ckpt_path = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/save/ckpts/st_pauls_cathedral/epoch=19.ckpt'
    use_rgb_loss = False
    main_create_aug(aug_dir, files, images_path, root_dir, ckpt_path, use_rgb_loss)
