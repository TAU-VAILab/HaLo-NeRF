import torch
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import imageio
from argparse import ArgumentParser
from models.rendering import render_rays
from models.nerf import *
from utils import load_ckpt
from datasets import dataset_dict
from datasets.depth_utils import *

from models.networks import E_attr
from PIL import Image
from sklearn.metrics import average_precision_score

torch.backends.cudnn.benchmark = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    parser.add_argument('--chunk', type=int, default=32 * 1024 * 2,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--ckpt_path', type=str, default='',
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


    # Flags For HaLo-NeRF (do not change) - ignore section
    parser.add_argument('--train_HaloNeRF_flag', default=False, action="store_true")
    parser.add_argument('--save_for_metric_flag', default=False, action="store_true")
    parser.add_argument('--calc_metrics_flag', default=False, action="store_true")
    parser.add_argument('--vis_flag', default=False, action="store_true")
    parser.add_argument('--prompts', type=str, default="spires;window;portal;facade")  # spires;window;portal;facade
    parser.add_argument('--top_k_files', type=int, default=150)
    parser.add_argument('--csv_retrieval_path', type=str,
                        default='data/ft_clip_sims_v0.2-ft_bsz128_5epochs-lr1e-06-val091-2430-notest24.csv')  #
    parser.add_argument('--path_gt', type=str,
                        default='data/manually_gt_masks_0_1/')  #
    parser.add_argument('--PRED_THRESHOLD', type=float, default=0.5)
    parser.add_argument('--save_training_vis', type=bool, default=True)
    parser.add_argument('--threshold', type=float, default=0.2)
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


def label_img_to_color(img):
    label_to_color = {
        0: [255, 255, 255],
        1: [244, 35, 232],
        2: [70, 70, 70],
        3: [102, 102, 156],
        4: [190, 153, 153],
        5: [153, 153, 153],
        6: [250, 170, 30],
        7: [220, 220, 0],
        8: [107, 142, 35],
        9: [152, 251, 152],
        10: [70, 130, 180],
        11: [220, 20, 60]
    }
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for key in label_to_color.keys():
        img_rgb[img == key] = label_to_color[key]

    return img_rgb.astype('uint8')


def label_img_to_blue(img):
    label_to_color = {
        0: [0, 0, 0],
        1: [0, 0, 255],
        2: [0, 0, 255],
        3: [0, 0, 255],
        4: [0, 0, 255],
        5: [0, 0, 255],
        6: [0, 0, 255],
        7: [0, 0, 255],
        8: [0, 0, 255],
        9: [0, 0, 255],
        10: [0, 0, 255],
        11: [0, 0, 255],
    }
    img_rgb = np.zeros((img.shape[0], img.shape[1], 3))

    for key in label_to_color.keys():
        img_rgb[img == key] = label_to_color[key]

    return img_rgb.astype('uint8')



def main_vis(save_training_vis, training_files, ts_list, root_dir, N_vocab, scene_name, ckpt_path, save_dir, clipseg_folder, path_gt, prompt, top_k, num_epochs):

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
    chunk = 10000

    kwargs = {'root_dir': root_dir,
              'split': split}

    kwargs['img_downscale'] = 2
    kwargs['use_cache'] = False


    dataset = dataset_dict[dataset_name](**kwargs)

    embedding_xyz = PosEmbedding(N_emb_xyz - 1, N_emb_xyz)
    embedding_dir = PosEmbedding(N_emb_dir - 1, N_emb_dir)
    embeddings = {'xyz': embedding_xyz, 'dir': embedding_dir}
    if encode_a:
        enc_a = E_attr(3, N_a).cuda()
        load_ckpt(enc_a, ckpt_path, model_name='enc_a')
        kwargs = {}

    nerf_coarse = NeRF('coarse',
                       enable_semantic=enable_semantic, num_semantic_classes=num_semantic_classes,
                       in_channels_xyz=6 * N_emb_xyz + 3,
                       in_channels_dir=6 * N_emb_dir + 3,
                       is_test=False).cuda()
    nerf_fine = NeRF('fine',
                     enable_semantic=enable_semantic, num_semantic_classes=num_semantic_classes,
                     in_channels_xyz=6 * N_emb_xyz + 3,
                     in_channels_dir=6 * N_emb_dir + 3,
                     encode_appearance=encode_a,
                     in_channels_a=N_a,
                     is_test=False).cuda()

    load_ckpt(nerf_coarse, ckpt_path, model_name='nerf_coarse')
    load_ckpt(nerf_fine, ckpt_path, model_name='nerf_fine')

    models = {'coarse': nerf_coarse, 'fine': nerf_fine}



    dir_name = os.path.join(save_dir, f'results/{dataset_name}/vis/top_{str(top_k)}_nEpochs{str(num_epochs)}/{scene_name}')
    os.makedirs(dir_name, exist_ok=True)
    os.makedirs(os.path.join(dir_name, 'train'), exist_ok=True)
    os.makedirs(os.path.join(dir_name, 'test'), exist_ok=True)

    dataset.test_img_w, dataset.test_img_h = img_wh
    imgs = []

    colormap = plt.get_cmap('jet')

    training_files = [int(f[:4]) for f in training_files]

    clipseg_folder_train = os.path.join('./sem_results', clipseg_folder)

    for i in tqdm(range(0, len(dataset), 1)):

        sample = dataset[i]
        rays = sample['rays']
        ts = sample['ts']

        is_ts_flag = ts[0] in ts_list
        is_training_flag = (ts[0] in training_files) and save_training_vis

        if (not is_ts_flag) and (not is_training_flag):
            continue

        if (split == 'test_train' or split == 'test_test') and encode_a:
            whole_img = sample['whole_img'].unsqueeze(0).cuda()
            kwargs['a_embedded_from_img'] = enc_a(whole_img)
        results = batched_inference(models, embeddings, rays.cuda(), ts.cuda(),
                                    N_samples, N_importance, use_disp,
                                    chunk,
                                    dataset.white_back,
                                    **kwargs)

        w, h = sample['img_wh']
        img_pred = np.clip(results['rgb_fine'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_pred_ = (img_pred * 255).astype(np.uint8)
        imgs += [img_pred_]
        # imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}.png'), img_pred_)

        img_GT = np.clip(sample['rgbs'].view(h, w, 3).cpu().numpy(), 0, 1)
        img_GT_ = (img_GT * 255).astype(np.uint8)
        # imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_GT_RGB.png'), img_GT_)


        sem_pred = results['semantics_fine'][:, 1].view(h, w, 1).cpu().numpy()
        sem_pred_original = sem_pred.squeeze()



        sem_pred = torch.Tensor(sem_pred.squeeze())
        imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_semantic.png'), sem_pred)

        heatmap = colormap(sem_pred).squeeze()

        h = heatmap[:, :, :3]
        sem_pred_original = np.expand_dims(sem_pred_original, axis=2)

        pred_with_overlay = 0.45 * np.multiply(1 - sem_pred_original, img_GT) + 0.55 * np.multiply(sem_pred_original,h)

        imageio.imwrite(os.path.join(dir_name, f'{ts[0]:03d}_semantic_jet.png'), heatmap)

        if is_ts_flag:
            clipseg_res = torch.load(os.path.join(path_gt, prompt, 'clipseg_results', str(int(ts[0])).zfill(4) + '.pickle'))

            clipseg_res = torch.nn.functional.interpolate(clipseg_res.unsqueeze(dim=0).unsqueeze(dim=0),
                                                           size=(img_GT_.shape[0], img_GT_.shape[1]), mode='bilinear').squeeze()

        elif is_training_flag:
            try:
                clipseg_res = torch.load(os.path.join(clipseg_folder_train, str(int(ts[0])).zfill(4) + '.pickle'))
                clipseg_res = torch.nn.functional.interpolate(clipseg_res.unsqueeze(dim=0).unsqueeze(dim=0),
                                                              size=(img_GT_.shape[0], img_GT_.shape[1]),
                                                              mode='bilinear').squeeze()
            except:
                clipseg_res = np.zeros_like(img_GT_)

        if is_ts_flag:
            try:
                sem_gt = imageio.imread(os.path.join(path_gt, prompt, str(int(ts[0])).zfill(4) + '_mask.jpg'))
                gt_for_metric = Image.open(os.path.join(path_gt, prompt, str(int(ts[0])).zfill(4) + '_mask.jpg')).convert('L')
                # sem_gt = imageio.imread(os.path.join(path_gt, prompt, z[j].replace('.pickle', '_mask.jpg')))
                # gt_for_metric = Image.open(os.path.join(path_gt, prompt, z[j].replace('.pickle', '_mask.jpg'))).convert('L')

            except:
                sem_gt = imageio.imread(os.path.join(path_gt, prompt, str(int(ts[0])).zfill(4) + '_mask.JPG'))
                gt_for_metric = Image.open(os.path.join(path_gt, prompt, str(int(ts[0])).zfill(4) + '_mask.JPG')).convert('L')
                # sem_gt = imageio.imread(os.path.join(path_gt, prompt, z[j].replace('.pickle', '_mask.JPG')))
                # gt_for_metric = Image.open(os.path.join(path_gt, prompt, z[j].replace('.pickle', '_mask.JPG'))).convert('L')


            # calculate metrics - clipseg
            gt_array = 1 - np.asarray(gt_for_metric) / 255
            gt_mask = gt_array > 0.5
            gt_mask_flat = gt_mask.ravel()
            clipseg_res_2 = torch.nn.functional.interpolate(clipseg_res.unsqueeze(dim=0).unsqueeze(dim=0),
                                                           size=(gt_mask.shape[0], gt_mask.shape[1]), mode='bilinear').squeeze()
            pred_array_flat = clipseg_res_2.numpy().ravel()
            AP_clipseg = average_precision_score(gt_mask_flat, pred_array_flat)

            # calculate metrics - sem pred
            sem_pred_2 = torch.nn.functional.interpolate(sem_pred.unsqueeze(dim=0).unsqueeze(dim=0),
                                                            size=(gt_mask.shape[0], gt_mask.shape[1]),
                                                            mode='bilinear').squeeze()
            pred_array_flat = sem_pred_2.numpy().ravel()
            AP_pred = average_precision_score(gt_mask_flat, pred_array_flat)

        # Plot
        if is_ts_flag:
            fig, axis = plt.subplots(1,5, figsize=(20,4))
            fig.suptitle(f'test data: {prompt}')
            axis[0].imshow(img_GT_)
            axis[0].title.set_text('rgb gt')
            axis[1].imshow(img_pred_)
            axis[1].title.set_text('rgb pred')
            axis[2].imshow(img_GT_)
            axis[2].imshow(clipseg_res, cmap=colormap, alpha=0.5)
            axis[2].title.set_text(f'clipseg pred, ap={AP_clipseg:.4f}')
            im = axis[3].imshow(pred_with_overlay)
            axis[3].title.set_text(f'sem pred, ap={AP_pred:.4f}')
            axis[4].imshow(sem_gt)
            axis[4].title.set_text('sem gt')
            for ax in axis:
                ax.axis('off')
            plt.tight_layout()
            fig.colorbar(im)
            fig.savefig(os.path.join(dir_name, 'test', f'{ts[0]:03d}_sem_fig.png'))
        elif is_training_flag:
            fig, axis = plt.subplots(1, 4, figsize=(20, 4))
            fig.suptitle(f'training data: {prompt}')
            axis[0].imshow(img_GT_)
            axis[0].title.set_text('rgb gt')
            axis[1].imshow(img_pred_)
            axis[1].title.set_text('rgb pred')
            axis[2].imshow(img_GT_)
            axis[2].imshow(clipseg_res, cmap=colormap, alpha=0.5)
            axis[2].title.set_text(f'clipseg pred')
            im = axis[3].imshow(pred_with_overlay)
            axis[3].title.set_text(f'sem pred')
            for ax in axis:
                ax.axis('off')
            plt.tight_layout()
            plt.colorbar(im)
            fig.savefig(os.path.join(dir_name, 'train', f'{ts[0]:03d}_sem_fig.png'))



if __name__ == "__main__":
    pass
    # ts_list = [1, 2]
    # main_eval(ts_list, root_dir, N_vocab, scene_name, ckpt_path)