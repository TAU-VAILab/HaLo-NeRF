import argparse
# from datasets import WikiScenesDataset
from datasets.phototourism_mask_grid_sample import PhototourismDataset
from utils.wikiscenes_utils import create_nerf_root_dir_from_ws
import numpy as np
import os
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True,
                        help='input directory of datatset (WikiScenes3D)')
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--img_downscale', type=int, default=2,
                        help='how much to modify width for WS dataset')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_opts()

    if not os.path.exists(args.root_dir):
        create_nerf_root_dir_from_ws(args.input_dir, args.root_dir)
        print('run: ' + 'LD_LIBRARY_PATH=/phoenix/S7/he93/Deploy/colmap/build/__install__/lib /phoenix/S7/he93/Deploy/colmap/build/__install__/bin/colmap image_undistorter --image_path images --input_path sparse --output_path undistorted --output_type COLMAP')
        exit()

    # args.root_dir = os.path.join(args.root_dir, 'dense', 'sparse')
    os.makedirs(os.path.join(args.root_dir, 'cache'), exist_ok=True)
    print(f'Preparing cache for img downsample {args.img_downscale}...')
    dataset = PhototourismDataset(args.root_dir, 'train', args.img_downscale)

    # save img ids
    with open(os.path.join(args.root_dir, f'cache/img_ids.pkl'), 'wb') as f:
        pickle.dump(dataset.img_ids, f, pickle.HIGHEST_PROTOCOL)
    # save img paths
    with open(os.path.join(args.root_dir, f'cache/image_paths.pkl'), 'wb') as f:
        pickle.dump(dataset.image_paths, f, pickle.HIGHEST_PROTOCOL)
    # save Ks
    with open(os.path.join(args.root_dir, f'cache/Ks{args.img_downscale}.pkl'), 'wb') as f:
        pickle.dump(dataset.Ks, f, pickle.HIGHEST_PROTOCOL)

    # save all_imgs
    with open(os.path.join(args.root_dir, f'cache/all_imgs{8}.pkl'), 'wb') as f:
        pickle.dump(dataset.all_imgs, f, pickle.HIGHEST_PROTOCOL)

    # save scene points
    np.save(os.path.join(args.root_dir, 'cache/xyz_world.npy'),
            dataset.xyz_world)
    # save poses
    np.save(os.path.join(args.root_dir, 'cache/poses.npy'),
            dataset.poses)
    # save near and far bounds
    with open(os.path.join(args.root_dir, f'cache/nears.pkl'), 'wb') as f:
        pickle.dump(dataset.nears, f, pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(args.root_dir, f'cache/fars.pkl'), 'wb') as f:
        pickle.dump(dataset.fars, f, pickle.HIGHEST_PROTOCOL)
    # save rays and rgbs
    np.save(os.path.join(args.root_dir, f'cache/rays{args.img_downscale}.npy'),
            dataset.all_rays.numpy())
    np.save(os.path.join(args.root_dir, f'cache/rgbs{args.img_downscale}.npy'),
            dataset.all_rgbs.numpy())

    # save all_imgs_wh
    np.save(os.path.join(args.root_dir, f'cache/all_imgs_wh{args.img_downscale}.npy'),
            dataset.all_imgs_wh.numpy())

    print(f"Data cache saved to {os.path.join(args.root_dir, 'cache')} !")