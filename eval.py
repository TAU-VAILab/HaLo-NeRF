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

if __name__ == "__main__":
    args = get_opts()

    if args.split == 'train':
        args.split = 'test_train'

    kwargs = {'root_dir': args.root_dir,
              'split': args.split}
    if args.dataset_name == 'blender':
        kwargs['img_wh'] = tuple(args.img_wh)
    else:
        kwargs['img_downscale'] = args.img_downscale
        kwargs['use_cache'] = args.use_cache

    dataset = dataset_dict[args.dataset_name](**kwargs)
    scene = os.path.basename(args.root_dir.strip('/'))

    embedding_xyz = PosEmbedding(args.N_emb_xyz-1, args.N_emb_xyz)
    embedding_dir = PosEmbedding(args.N_emb_dir-1, args.N_emb_dir) 
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if args.encode_a:
        # enc_a
        enc_a = E_attr(3, args.N_a).cuda()
        load_ckpt(enc_a, args.ckpt_path, model_name='enc_a')
        kwargs = {}


    nerf_coarse = NeRF('coarse',
                       enable_semantic=args.enable_semantic, num_semantic_classes=args.num_semantic_classes,
                        in_channels_xyz=6*args.N_emb_xyz+3,
                        in_channels_dir=6*args.N_emb_dir+3,
                        is_test=True).cuda()
    nerf_fine = NeRF('fine',
                     enable_semantic=args.enable_semantic, num_semantic_classes=args.num_semantic_classes,
                     in_channels_xyz=6*args.N_emb_xyz+3,
                     in_channels_dir=6*args.N_emb_dir+3,
                     encode_appearance=args.encode_a,
                     in_channels_a=args.N_a,
                     is_test=True).cuda()

    load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, args.ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}


    if args.enable_semantic:

        nerf_coarse_sem = NeRF('coarse',
                           enable_semantic=args.enable_semantic, num_semantic_classes=args.num_semantic_classes,
                            in_channels_xyz=6*args.N_emb_xyz+3,
                            in_channels_dir=6*args.N_emb_dir+3,
                            is_test=False).cuda()
        nerf_fine_sem = NeRF('fine',
                         enable_semantic=args.enable_semantic, num_semantic_classes=args.num_semantic_classes,
                         in_channels_xyz=6*args.N_emb_xyz+3,
                         in_channels_dir=6*args.N_emb_dir+3,
                         encode_appearance=args.encode_a,
                         in_channels_a=args.N_a,
                         is_test=False).cuda()

        load_ckpt(nerf_coarse_sem, args.ckpt_path, model_name='nerf_coarse')
        load_ckpt(nerf_fine_sem, args.ckpt_path, model_name='nerf_fine')
        models_sem = {'coarse': nerf_coarse_sem, 'fine': nerf_fine_sem}




    dir_name = os.path.join(args.save_dir, f'results/{args.dataset_name}/{args.scene_name}')
    os.makedirs(dir_name, exist_ok=True)

    dataset.test_img_w, dataset.test_img_h = args.img_wh
    imgs = []
    sem_preds = []
    preds_with_overlay = []

    # define testing poses and appearance index for phototourism
    if args.dataset_name == 'phototourism' and args.split == 'test':
        # define testing camera intrinsics (hard-coded, feel free to change)
        dataset.test_img_w, dataset.test_img_h = args.img_wh
        dataset.test_focal = dataset.test_img_w/2/np.tan(np.pi/6) # fov=60 degrees
        dataset.test_K = np.array([[dataset.test_focal, 0, dataset.test_img_w/2],
                                   [0, dataset.test_focal, dataset.test_img_h/2],
                                   [0,                  0,                    1]])


        img = Image.open(os.path.join(args.root_dir, 'dense/images',
                                      dataset.image_paths[dataset.img_ids_train[args.images_id_appearance_first]])).convert('RGB')  # 111 159 178 208 252 314
        img_downscale = 8
        img_w, img_h = img.size
        img_w = img_w // img_downscale
        img_h = img_h // img_downscale
        img = img.resize((img_w, img_h), Image.LANCZOS)
        toTensor = T.ToTensor()
        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        img = toTensor(img)  # (3, h, w)
        whole_img = normalize(img).unsqueeze(0).cuda()
        kwargs['a_embedded_from_img'] = enc_a(whole_img)

        dataset.test_appearance_idx = args.images_id_appearance_first  # 85572957_6053497857.jpg
        N_frames = 30

        dx = np.linspace(-0.1, 0.1, N_frames)

        dy1 = np.linspace(-0., 0, N_frames // 2)  # + down
        dy2 = np.linspace(-0., 0, N_frames - N_frames // 2)
        dy = np.concatenate((dy1, dy2))

        dz = np.linspace(-0, 0.3, N_frames)

        theta_x = np.linspace(0, 0, N_frames)
        theta_y = np.linspace(0, 0, N_frames)
        theta_z = np.linspace(0, 0, N_frames)
        # define poses
        dataset.poses_test = np.tile(dataset.poses_dict[6], (N_frames, 1, 1))
        for i in range(N_frames):
            dataset.poses_test[i, 0, 3] += dx[i]
            dataset.poses_test[i, 1, 3] += dy[i]
            dataset.poses_test[i, 2, 3] += dz[i]
            dataset.poses_test[i, :, :3] = np.dot(eulerAnglesToRotationMatrix([theta_x[i],theta_y[i],theta_z[i]]), dataset.poses_test[i, :, :3])


        dataset.poses_test = generate_camera_path(dataset, args.images_ids, args.num_frames)



    kwargs['output_transient'] = False
    colormap = plt.get_cmap('jet')

    # sample_enc = dataset[1]
    # whole_img_enc = sample_enc['whole_img'].unsqueeze(0).cuda()
    for i in tqdm(range(0,len(dataset),1)):

        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']

        if (args.split == 'test_train' or args.split == 'test_test') and args.encode_a:
            whole_img = sample['whole_img'].unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
            # kwargs['a_embedded_from_img'] = enc_a(whole_img_enc)


        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    args.N_samples, args.N_importance, args.use_disp,
                                    args.chunk,
                                    dataset.white_back,
                                    **kwargs)
        w, h = sample['img_wh']


        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred*255).astype(np.uint8)
        imgs += [img_pred_]

        if args.save_imgs:
            imageio.imwrite(os.path.join(dir_name, f'{i:03d}.png'), img_pred_)


        if args.enable_semantic:
            sem_pred_original = results['semantics_fine'][:,1].view(h, w, 1).cpu().numpy()
            sem_pred = (sem_pred_original * 255).astype(np.uint8)



            sem_pred_colormap = colormap(sem_pred).squeeze()
            sem_preds += [sem_pred_colormap]

            if args.save_imgs:
                imageio.imwrite(os.path.join(dir_name, f'{i:03d}_semantic_jet.png'), sem_pred_colormap)


            if args.save_imgs:
                imageio.imwrite(os.path.join(dir_name, f'{i:03d}_semantic.png'), sem_pred_original)

                if args.blur_pred:
                    sem_pred = cv2.imread(os.path.join(dir_name, f'{i:03d}_semantic.png'), cv2.IMREAD_GRAYSCALE)
                    w, h = sem_pred.shape
                    m = min(w, h)
                    r = 500 / m
                    sem_pred = cv2.resize(sem_pred, (int(h * r), int(w * r)))
                    sem_pred = cv2.GaussianBlur(sem_pred, (11, 11), 10)
                    sem_pred = cv2.resize(sem_pred, (h, w))

                C = np.zeros((256, 4), dtype=np.float32)
                C[:, 1] = 1.
                C[:, -1] = np.linspace(0, 1, 256)
                cmap_ = ListedColormap(C)
                x = 100 / 1.3

                w, h, _ = img_pred_.shape
                figsize = h / float(x), w / float(x)
                fig = plt.figure(figsize=figsize)
                plt.imshow(img_pred_)
                sem_pred_ = sem_pred_original.squeeze()
                plt.imshow(sem_pred_, cmap=cmap_, alpha=np.asarray(sem_pred_))
                plt.axis('off')
                plt.savefig(os.path.join(dir_name, f'{i:03d}_semantic_green.png'), bbox_inches="tight",
                            pad_inches=0)
                plt.close()


                if args.save_real_images_semantic:
                    try:
                        images_path = os.path.join(args.root_dir, 'dense/images', str(int(ts[0])).zfill(4) + '.jpg')
                        real_img = Image.open(images_path)
                    except:
                        try:
                            print('JPG File')
                            images_path = os.path.join(args.root_dir, 'dense/images', str(int(ts[0])).zfill(4) + '.JPG')
                            real_img = Image.open(images_path)
                        except:
                            images_path = os.path.join(args.root_dir, 'dense/images', str(int(ts[0])).zfill(4) + '.jpeg')
                            real_img = Image.open(images_path)

                    real_img = real_img.resize((h, w))
                    fig = plt.figure(figsize=figsize)
                    plt.imshow(real_img)
                    plt.imshow(sem_pred_, cmap=cmap_, alpha=np.asarray(sem_pred_))
                    plt.axis('off')
                    plt.savefig(os.path.join(dir_name, f'{i:03d}_semantic_green_real.png'), bbox_inches="tight",
                                pad_inches=0)
                    plt.close()

        if args.split == 'test':
            imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_rgb.{args.video_format}'),imgs, fps=24)
            if args.enable_semantic:
                imageio.mimsave(os.path.join(dir_name, f'{args.scene_name}_sem.{args.video_format}'),sem_preds, fps=24)

    print('Done')

