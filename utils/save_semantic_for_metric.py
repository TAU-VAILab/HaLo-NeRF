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
    parser.add_argument('--split', type=str, default='test_train',
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
    parser.add_argument('--N_vocab', type=int, default=100,
                        help='''number of vocabulary (number of images) 
                                in the dataset for nn.Embedding''')
    parser.add_argument('--encode_a', default=True, action="store_true",
                        help='whether to encode appearance')
    parser.add_argument('--N_a', type=int, default=48,
                        help='number of embeddings for appearance')

    parser.add_argument('--chunk', type=int, default=32*1024*1,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, default= '',
                        help='pretrained checkpoint path to load')

    parser.add_argument('--video_format', type=str, default='mp4',
                        choices=['gif', 'mp4'],
                        help='video format, gif or mp4')
    
    parser.add_argument('--save_dir', type=str, default="./save",
                        help='pretrained checkpoint path to load')

    ## Semantic Nerf
    parser.add_argument('--enable_semantic', default=True, action="store_true",
                        help='whether to enable semantics')
    parser.add_argument('--num_semantic_classes', type=int, default=2,
                        help='The number of semantic classes')



    # Flags For HaloNeRF (do not change)
    parser.add_argument('--train_HaloNeRF_flag', default=False, action="store_true")
    parser.add_argument('--save_for_metric_flag', default=False, action="store_true")
    parser.add_argument('--calc_metrics_flag', default=False, action="store_true")
    parser.add_argument('--vis_flag', default=False, action="store_true")
    parser.add_argument('--prompts', type=str, default="spires;window;portal;facade")  # spires;window;portal;facade
    parser.add_argument('--top_k_files', type=int, default=15)
    parser.add_argument('--xls_path', type=str,
                        default='/home/cc/students/csguests/chendudai/Thesis/data/ft_clip_sims_v0.2-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24.csv')  #
    parser.add_argument('--use_rgb_loss', type=bool, default=False)
    parser.add_argument('--path_gt', type=str,
                        default='/home/cc/students/csguests/chendudai/Thesis/data/manually_gt_masks_0_1/')  #
    parser.add_argument('--PRED_THRESHOLD', type=float, default=0.5)
    parser.add_argument('--save_training_vis', type=bool, default=True)
    parser.add_argument('--neg_prec', type=float, default=0.1)
    parser.add_argument('--use_threshold', type=bool, default=True)
    parser.add_argument('--threshold', type=float, default=0.2)
    parser.add_argument('--vis_prompt_path', type=str, default='/home/cc/students/csguests/chendudai/Thesis/data/visual_prompts_v1.pk')
    parser.add_argument('--use_vis_prompt', default=False, action="store_true")
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')



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
                        rays[i:i+chunk],
                        ts[i:i+chunk] if ts is not None else None,
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


def label_img_to_color(img):
    label_to_color = {
    0: [255, 255,255],
    1: [244, 35,232],
    2: [ 70, 70, 70],
    3: [102,102,156],
    4: [190,153,153],
    5: [153,153,153],
    6: [250,170, 30],
    7: [220,220,  0],
    8: [107,142, 35],
    9: [152,251,152],
    10: [ 70,130,180],
    11: [220, 20, 60]
    }
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for key in label_to_color.keys():
        img_rgb[img == key] = label_to_color[key]

    return img_rgb.astype('uint8')

def label_img_to_blue(img):
    label_to_color = {
    0: [0, 0,0],
    1: [0, 0,255],
    2: [0, 0,255],
    3: [0, 0,255],
    4: [0, 0,255],
    5: [0, 0,255],
    6: [0, 0,255],
    7: [0, 0,255],
    8: [0, 0,255],
    9: [0, 0,255],
    10: [0, 0,255],
    11: [0, 0,255],
    }
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for key in label_to_color.keys():
        img_rgb[img == key] = label_to_color[key]

    return img_rgb.astype('uint8')

def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def main_eval(ts_list, root_dir, N_vocab, scene_name, ckpt_path, save_dir, top_k, num_epochs):

    # args = get_opts()
    # args.root_dir = root_dir
    # args.N_vocab = N_vocab
    # args.scene_name = scene_name
    # args.ckpt_path = ckpt_path
    # args.save_dir = save_dir
    N_a = 48
    N_emb_xyz = 15
    N_emb_dir = 4
    encode_a = True
    dataset_name = 'phototourism'
    img_wh = [500,500]
    split = 'test_train'
    enable_semantic = True
    num_semantic_classes = 2
    N_importance = 256
    N_samples = 256
    use_disp = False
    chunk = 32768

    kwargs = {'root_dir': root_dir,
              'split': split}

    kwargs['img_downscale'] = 2
    kwargs['use_cache'] = False


    dataset = dataset_dict[dataset_name](**kwargs)

    # kwargs['split'] = 'test_test'
    #
    # dataset_test = dataset_dict[args.dataset_name](**kwargs)


    embedding_xyz = PosEmbedding(N_emb_xyz-1, N_emb_xyz)
    embedding_dir = PosEmbedding(N_emb_dir-1, N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if encode_a:
        # enc_a
        enc_a = E_attr(3, N_a).cuda()
        load_ckpt(enc_a, ckpt_path, model_name='enc_a')
        kwargs = {}


    nerf_coarse = NeRF('coarse',
                       enable_semantic=enable_semantic, num_semantic_classes=num_semantic_classes,
                        in_channels_xyz=6*N_emb_xyz+3,
                        in_channels_dir=6*N_emb_dir+3).cuda()
    nerf_fine = NeRF('fine',
                     enable_semantic=enable_semantic, num_semantic_classes=num_semantic_classes,
                     in_channels_xyz=6*N_emb_xyz+3,
                     in_channels_dir=6*N_emb_dir+3,
                     encode_appearance=encode_a,
                     in_channels_a=N_a).cuda()

    load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}

    dir_name = os.path.join(save_dir, f'results/{dataset_name}/for_metric/top_{str(top_k)}_nEpochs{str(num_epochs)}/{scene_name}')
    os.makedirs(dir_name, exist_ok=True)

    dataset.test_img_w, dataset.test_img_h = img_wh


    for i in tqdm(range(0,len(dataset),1)):
        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']

        if ts[0] not in ts_list:
            continue

        # ts = torch.zeros(len(rays), dtype=torch.long)
        if (split == 'test_train' or split == 'test_test') and encode_a:
            whole_img = sample['whole_img'].unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    N_samples, N_importance, use_disp,
                                    chunk,
                                    dataset.white_back,
                                    **kwargs)

        w, h = sample['img_wh']

        try:
            images_path = os.path.join(root_dir, 'dense/images',str(int(ts[0])).zfill(4) + '.jpg')
            real_img = Image.open(images_path)
        except:
            print('JPG File')
            images_path = os.path.join(root_dir, 'dense/images',str(int(ts[0])).zfill(4) + '.JPG')
            real_img = Image.open(images_path)

        real_w, real_h = real_img.size

        sem_pred = results['semantics_fine'][:,1].view(h, w, 1).cpu().numpy()
        sem_pred = 1 - sem_pred
        sem_pred = torch.Tensor(sem_pred.squeeze())
        sem_pred = torch.nn.functional.interpolate(sem_pred.unsqueeze(dim=0).unsqueeze(dim=0),
                                                       size=(real_h, real_w), mode='bilinear').squeeze().numpy()
        sem_pred_ = (sem_pred * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_semantic.png'), sem_pred_)





    #
    # for i in tqdm(range(0, len(dataset_test), 1)):
    #
    #     sample = dataset_test[i]
    #     rays = sample['rays']
    #     ts = sample['ts']
    #
    #     if ts[0] not in ts_list:
    #         continue
    #
    #     # ts = torch.zeros(len(rays), dtype=torch.long)
    #     if (args.split == 'test_train' or args.split == 'test_test') and args.encode_a:
    #         whole_img = sample['whole_img'].unsqueeze(0).cuda()
    #         kwargs['a_embedded_from_img'] = enc_a(whole_img)
    #     results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
    #                                 args.N_samples, args.N_importance, args.use_disp,
    #                                 args.chunk,
    #                                 dataset_test.white_back,
    #                                 **kwargs)
    #
    #     w, h = sample['img_wh']
    #
    #     try:
    #         images_path = os.path.join(args.root_dir, 'dense/images', str(int(ts[0])).zfill(4) + '.jpg')
    #         real_img = Image.open(images_path)
    #     except:
    #         print('JPG File')
    #         images_path = os.path.join(args.root_dir, 'dense/images', str(int(ts[0])).zfill(4) + '.JPG')
    #         real_img = Image.open(images_path)
    #
    #     real_w, real_h = real_img.size
    #
    #     sem_pred = results['semantics_fine'][:, 1].view(h, w, 1).cpu().numpy()
    #     sem_pred = 1 - sem_pred
    #     sem_pred = torch.Tensor(sem_pred.squeeze())
    #     sem_pred = torch.nn.functional.interpolate(sem_pred.unsqueeze(dim=0).unsqueeze(dim=0),
    #                                                size=(real_h, real_w), mode='bilinear').squeeze().numpy()
    #     sem_pred_ = (sem_pred * 255).astype(np.uint8)
    #     imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_semantic.png'), sem_pred_)

if __name__ == "__main__":
    pass
    # ts_list = [1, 2]
    # main_eval(ts_list, root_dir, N_vocab, scene_name, ckpt_path)