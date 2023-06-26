import os
from datasets.colmap_utils import read_model, write_model #  replace ht.lib.colmap with colmap from "https://github.com/colmap/colmap/tree/dev/scripts/python"
import trimesh
import numpy as np
import tqdm
import copy
import torch

def main(input_sparse, input_ply, output_path, thresh):
    '''
    Generate the list of interested images from a COLMAP sparse model and ply.
    Specifically, select images that rely on the point cloud of interest to complete the registration.
    '''
    torch.cuda.empty_cache()

    cameras, images, points3D = read_model(input_sparse, '.bin')
    roi_ply = trimesh.load_mesh(input_ply)
    roi_points = np.array(roi_ply.vertices)
    points = np.array([points3D[k].xyz for k in points3D])
    points_id = np.array([points3D[k].id for k in points3D])

    # Generate points_id for the interested point cloud
    roi_points_id = []
    chunk = 512 # Reduce it if CUDA out of memory occurs
    roi_points_cuda = torch.tensor(roi_points).cuda().float()
    points_cuda = torch.tensor(points).cuda().float()
    for i in tqdm.tqdm(range(0, roi_points.shape[0], chunk)):
        chunk_points = roi_points_cuda[i:i+chunk]
        distance = (chunk_points[:, None] - points_cuda[None]).norm(dim=-1)
        roi_points_id += points_id[distance.argmin(dim=-1).cpu().numpy()].tolist()
    roi_points_id = np.array(roi_points_id)

    # Select interested images
    save_images = []
    save_conds = []
    for im_id, image in tqdm.tqdm(images.items()):
        im_points_ids = image.point3D_ids[image.point3D_ids!=-1]
        save_conds.append(np.isin(im_points_ids, roi_points_id).mean())
        save_images.append(image.name)
    save_list = np.array(save_images)[(np.array(save_conds) > 0.8)].tolist()
    print('Interested: {}'.format(len(save_list)))

    open(output_path, 'w').writelines([item + '\n' for item in save_list])

    # images_new = {}
    # for i, image in enumerate(images):
    #     if images[i].name in save_list:
    #         images_new[i] = images[i]
    #
    # points3D_new = {}
    # for i, point in enumerate(roi_points_id):
    #     points3D_new[roi_points_id[i]] = points3D[roi_points_id[i]]
    #
    # # images_new = dict(images_new)
    # write_model(cameras, images_new, points3D_new, path='/mnt/data/chendudai/repos/HaNeRF/data/WikiScenes/98_3_undistorted', ext=".bin")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_sparse', type=str, help="the path of COLMAP input sparse model")
    parser.add_argument('--input_ply', type=str, help="the path of interested point cloud (Manually filtered from the point cloud of the sparse model.)")
    parser.add_argument('--output_path', type=str, help="output list file")
    parser.add_argument('--thresh', type=float, default=0.8)
    args = parser.parse_args()
    main(args.input_sparse, args.input_ply, args.output_path, args.thresh)
