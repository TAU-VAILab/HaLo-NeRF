import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
from torchvision import transforms as T

from matplotlib import pyplot as plt
from .ray_utils import *
from .colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary

from math import sqrt, exp
import random
import imageio
from torchvision import transforms
from . import global_val
# from skimage.transform import resize
from pathlib import Path

class ImageData:
    def __init__(self, id, name, camera_id):
        self.id = id
        self.name = name
        self.camera_id = camera_id

class PhototourismDataset(Dataset):
    def __init__(self, root_dir, split='train', img_downscale=1, val_num=1, use_cache=False, batch_size=1024, scale_anneal=-1, min_scale=0.25, semantics_dir=[], files_to_run=[], neg_files=[], use_semantic_function='', threshold=0.2):
        """
        img_downscale: how much scale to downsample the training images.
                       The original image sizes are around 500~100, so value of 1 or 2
                       are recommended.
                       ATTENTION! Value of 1 will consume large CPU memory,
                       about 40G for brandenburg gate.
        val_num: number of val images (used for multigpu, validate same image for all gpus)
        use_cache: during data preparation, use precomputed rays (useful to accelerate
                   data loading, especially for multigpu!)
        """
        self.files_to_run = files_to_run
        self.neg_files = neg_files
        self.use_semantic_function = use_semantic_function
        self.semantics_dir = semantics_dir
        self.root_dir = root_dir
        self.split = split
        self.threshold = threshold

        assert img_downscale >= 1, 'image can only be downsampled, please set img_downscale>=1!'
        self.img_downscale = img_downscale

        if ('hagia_sophia_interior' in self.root_dir) or ('taj_mahal' in self.root_dir):
            self.img_downscale_appearance = 4
        else:
            # self.img_downscale_appearance = 1
            self.img_downscale_appearance = 4

        if split == 'val': # image downscale=1 will cause OOM in val mode
            self.img_downscale = max(1, self.img_downscale)
        self.val_num = max(1, val_num) # at least 1
        self.use_cache = use_cache
        self.define_transforms()

        self.read_meta()
        self.white_back = False

        # no effect if scale_anneal<0, else the minimum scale decreases exponentially until converge to min_scale
        self.scale_anneal = scale_anneal
        self.min_scale = min_scale

        self.batch_size = batch_size


    def read_meta_clipseg(self):
        path = '/home/cc/students/csguests/chendudai/Thesis/repos/Ha-NeRF/save/results/phototourism/0_1_frontview/'
        self.path = path
        cat = 'windows'
        files = os.listdir(path)
        self.poses = []
        self.files = []
        self.image_paths = []
        imdata = []
        i = 0
        for f in files:
            if f[-7:] == '.pickle':
                full_path = os.path.join(path, f)
                with open(full_path, 'rb') as handle:
                    img_data = pickle.load(handle)
                self.poses += [img_data['poses']]
                data = ImageData(i, full_path, i)
                imdata.append(data)
                self.image_paths += [os.path.join(path, f.split('_')[0]+ '.png')]
                i = i + 1

        self.poses_dict = {i: self.poses[i] for i, id_ in enumerate(self.poses)}
        self.N_images_train = len((self.poses))
        self.N_images_test = 0
        self.img_ids_train = range(0, self.N_images_train,1)
        self.img_ids_test = []
        self.img_ids = self.img_ids_train


        if  self.split in ['train']:

            imdata = []
            i = 0
            self.all_rays = []
            self.all_rgbs = []
            self.all_semantics_gt = []
            self.all_imgs = []
            self.all_imgs_wh = []

            for i, f in enumerate(files):
                if f[-7:] == '.pickle':
                    full_path = os.path.join(path, f)
                    with open(full_path, 'rb') as handle:
                        img_data = pickle.load(handle)

                    sample = img_data['sample']

                    sample['ts'] = sample['ts'] + i # TODO: Change to the eval creating data part


                    self.all_rays += [torch.cat((sample['rays'], sample['ts'].unsqueeze(dim=1)), dim=1)]


                    [img_w, img_h, c] = img_data['rgb'].shape
                    self.all_rgbs += [torch.Tensor(img_data['rgb'].reshape([img_w * img_h, 3])) / 255]


                    semantic_file = f.split('_')[0] + '.pickle'
                    with open(os.path.join(path, cat, semantic_file), 'rb') as handle:
                        semantics_gt = torch.load(handle)
                    semantics_gt = torch.nn.functional.interpolate(semantics_gt.unsqueeze(dim=0).unsqueeze(dim=0),
                                                                       size=(img_h, img_w))
                    semantics_gt = semantics_gt.reshape(-1,)
                    self.all_semantics_gt += [semantics_gt]
                    self.all_imgs += [img_data['whole_img'].squeeze(dim=0)]
                    self.all_imgs_wh += [torch.Tensor([img_w, img_h]).unsqueeze(0)]
                    self.poses += [img_data['poses']]


            self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 8)
            self.all_semantics_gt = torch.cat(self.all_semantics_gt, 0) # ((N_images-1)*h*w, 8)
            self.all_imgs_wh = torch.cat(self.all_imgs_wh, 0)  # ((N_images-1)*h*w, 3)




        elif self.split in ['val', 'test_train']:
            self.val_id = 0
        else:
            pass


    def read_meta(self):
        # read all files in the tsv first (split to train and test later)
        tsv = glob.glob(os.path.join(self.root_dir, '*.tsv'))[0]
        self.scene_name = os.path.basename(tsv)[:-4]
        self.files = pd.read_csv(tsv, sep='\t')
        self.files = self.files[~self.files['id'].isnull()] # remove data without id


        self.files['split'] = self.files['split'].replace('test', 'train')



        self.files.reset_index(inplace=True, drop=True)

        # Step 1. load image paths
        # Attention! The 'id' column in the tsv is BROKEN, don't use it!!!!
        # Instead, read the id from images.bin using image file name!
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/img_ids.pkl'), 'rb') as f:
                self.img_ids = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/image_paths.pkl'), 'rb') as f:
                self.image_paths = pickle.load(f)
        else:
            imdata = read_images_binary(os.path.join(self.root_dir, 'dense/sparse/images.bin'))

            # with open(os.path.join(self.root_dir, 'res.txt'), 'rb') as f:
            #     images_selected = f.readlines()

            img_path_to_id = {}
            for v in imdata.values():
                img_path_to_id[v.name] = v.id
            self.img_ids = []
            self.image_paths = {} # {id: filename}
            self.cam_ids = []

            # Convert to the correct format
            # for j,i in enumerate(images_selected):
            #     images_selected[j] = i[0:8].decode("utf-8")

            for filename in list(self.files['filename']):
                # if filename in img_path_to_id and filename in images_selected:
                if filename in img_path_to_id:
                    id_ = img_path_to_id[filename]
                    self.image_paths[id_] = filename
                    self.img_ids += [id_]
                    self.cam_ids += [imdata[id_].camera_id]

        # Step 2: read and rescale camera intrinsics
        if self.use_cache:
            with open(os.path.join(self.root_dir, f'cache/Ks{self.img_downscale}.pkl'), 'rb') as f:
                self.Ks = pickle.load(f)
        else:
            self.Ks = {} # {id: K}
            camdata = read_cameras_binary(os.path.join(self.root_dir, 'dense/sparse/cameras.bin'))
            for i,id_ in enumerate(self.img_ids):
                K = np.zeros((3, 3), dtype=np.float32)
                cam = camdata.get(self.cam_ids[i])
                if cam is not None:

                    img_w, img_h = int(cam.params[2]*2), int(cam.params[3]*2)

                    # I added *2 if the img_downscale = 1 but accutaly the image was downscaled before that.
                    img_w_, img_h_ = img_w//(self.img_downscale*1), img_h//(self.img_downscale*1)


                    K[0, 0] = cam.params[0]*img_w_/img_w # fx
                    K[1, 1] = cam.params[1]*img_h_/img_h # fy
                    K[0, 2] = cam.params[2]*img_w_/img_w # cx
                    K[1, 2] = cam.params[3]*img_h_/img_h # cy
                    K[2, 2] = 1
                    self.Ks[id_] = K

                else:
                    print('Error:' + str(id_))

        # Step 3: read c2w poses (of the images in tsv file only) and correct the order
        if self.use_cache:
            self.poses = np.load(os.path.join(self.root_dir, 'cache/poses.npy'))
        else:
            w2c_mats = []
            bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
            for id_ in self.img_ids:
                im = imdata[id_]
                R = im.qvec2rotmat()
                t = im.tvec.reshape(3, 1)
                w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
            w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
            self.poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
            # Original poses has rotation in form "right down front", change to "right up back"
            self.poses[..., 1:3] *= -1

        # Step 4: correct scale
        if self.use_cache:
            self.xyz_world = np.load(os.path.join(self.root_dir, 'cache/xyz_world.npy'))
            with open(os.path.join(self.root_dir, f'cache/nears.pkl'), 'rb') as f:
                self.nears = pickle.load(f)
            with open(os.path.join(self.root_dir, f'cache/fars.pkl'), 'rb') as f:
                self.fars = pickle.load(f)
        else:
            pts3d = read_points3d_binary(os.path.join(self.root_dir, 'dense/sparse/points3D.bin'))
            self.xyz_world = np.array([pts3d[p_id].xyz for p_id in pts3d])
            xyz_world_h = np.concatenate([self.xyz_world, np.ones((len(self.xyz_world), 1))], -1)
            # Compute near and far bounds for each image individually
            self.nears, self.fars = {}, {} # {id_: distance}
            for i, id_ in enumerate(self.img_ids):
                xyz_cam_i = (xyz_world_h @ w2c_mats[i].T)[:, :3] # xyz in the ith cam coordinate
                xyz_cam_i = xyz_cam_i[xyz_cam_i[:, 2]>0] # filter out points that lie behind the cam
                self.nears[id_] = np.percentile(xyz_cam_i[:, 2], 0.1)
                self.fars[id_] = np.percentile(xyz_cam_i[:, 2], 99.9)

            max_far = np.fromiter(self.fars.values(), np.float32).max()
            scale_factor = max_far/5 # so that the max far is scaled to 5
            self.poses[..., 3] /= scale_factor
            for k in self.nears:
                self.nears[k] /= scale_factor
            for k in self.fars:
                self.fars[k] /= scale_factor
            self.xyz_world /= scale_factor
        self.poses_dict = {id_: self.poses[i] for i, id_ in enumerate(self.img_ids)}
            
        # Step 5. split the img_ids (the number of images is verfied to match that in the paper)
        self.img_ids_train = [id_ for i, id_ in enumerate(self.img_ids) 
                                    if self.files.loc[i, 'split']=='train']
        self.img_ids_test = [id_ for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.img_names_test = [self.files.loc[i, 'filename'] for i, id_ in enumerate(self.img_ids)
                                    if self.files.loc[i, 'split']=='test']
        self.img_names_train = [self.files.loc[i, 'filename'] for i, id_ in enumerate(self.img_ids)
                               if self.files.loc[i, 'split'] == 'train']
        self.N_images_train = len(self.img_ids_train)
        self.N_images_test = len(self.img_ids_test)

        if self.split == 'train': # create buffer of all rays and rgb data
            if self.use_cache:
                all_rays = np.load(os.path.join(self.root_dir,
                                                f'cache/rays{self.img_downscale}.npy'))
                self.all_rays = torch.from_numpy(all_rays)
                all_rgbs = np.load(os.path.join(self.root_dir,
                                                f'cache/rgbs{self.img_downscale}.npy'))
                self.all_rgbs = torch.from_numpy(all_rgbs)
                # with open(os.path.join(self.root_dir, f'cache/all_imgs{self.img_downscale}.pkl'), 'rb') as f:
                #     self.all_imgs = pickle.load(f)
                with open(os.path.join(self.root_dir, f'cache/all_imgs{8}.pkl'), 'rb') as f:
                    self.all_imgs = pickle.load(f)
                all_imgs_wh = np.load(os.path.join(self.root_dir,
                                                f'cache/all_imgs_wh{self.img_downscale}.npy'))
                self.all_imgs_wh = torch.from_numpy(all_imgs_wh)
            else:
                self.all_rays = []
                self.all_rgbs = []
                self.all_imgs = []
                self.all_imgs_wh = []
                self.all_semantics_gt = []
                q = 0

                if self.files_to_run != []:
                    files_to_run = [int(f[:4]) for f in self.files_to_run]
                if self.neg_files != []:
                    neg_files = [int(f[:4]) for f in self.neg_files]

                is_neg = False
                is_pos = False
                for id_ in self.img_ids_train:
                    print(q)
                    q = q + 1

                    if self.neg_files != [] or self.files_to_run != []:

                        is_neg = (self.neg_files != [] and id_ in neg_files)
                        is_pos = (self.files_to_run != [] and id_ in files_to_run)

                        if not is_neg and not is_pos:
                            continue



                    c2w = torch.FloatTensor(self.poses_dict[id_])

                    img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                  self.image_paths[id_])).convert('RGB')


                    img_w, img_h = img.size






                    if self.img_downscale >= 1:
                        img_w = img_w//self.img_downscale
                        img_h = img_h//self.img_downscale
                        img_rs = img.resize((img_w, img_h), Image.LANCZOS)

                    semantics_gt = torch.Tensor([np.zeros(img_rs.size)]).squeeze(dim=0)
                    # semantics_dir = os.path.join(self.root_dir, 'dense/semantics')
                    # semantics_dir = '/storage/chendudai/data/0_1_undistorted/dense/door/'

                    if (self.semantics_dir != []) and not is_neg:
                        path_semantics = os.path.join(self.semantics_dir, self.image_paths[id_].split('.')[0]) + '.pickle'

                        if os.path.exists(path_semantics):
                            print(path_semantics)
                            try:
                                with open(path_semantics, 'rb') as f:
                                    semantics_gt = torch.Tensor(torch.load(f))
                            except:
                                with open(path_semantics, 'rb') as f:
                                    semantics_gt = torch.Tensor(pickle.load(f))

                                if self.use_semantic_function != '':
                                    if self.use_semantic_function == 'sigmoid':
                                        semantics_gt = 1 / (1 + torch.exp(-15 * (semantics_gt - 0.2)))
                                    elif self.use_semantic_function == 'double':
                                        semantics_gt = semantics_gt ** 2
                                    elif self.use_semantic_function == 'triple':
                                        semantics_gt = semantics_gt ** 3
                                    else:
                                        ValueError('no semantic function')

                            semantics_gt = torch.nn.functional.interpolate(semantics_gt.unsqueeze(dim=0).unsqueeze(dim=0),
                                                                           size=(img_h, img_w))

                            semantics_gt = semantics_gt.squeeze(dim=0).permute(1, 2, 0)

                            semantics_gt[semantics_gt < self.threshold] = 0.01
                            semantics_gt[semantics_gt >= self.threshold] = 1


                        else:
                            print('no sem path')


                    semantics_gt = semantics_gt.reshape(-1,)

                    img_w, img_h = img_rs.size
                    img_rs = self.transform(img_rs)  # (3, h, w)



                    img_8 = img.resize((img_w//self.img_downscale_appearance, img_h//self.img_downscale_appearance), Image.LANCZOS)


                    const_minSize = 33
                    if img_8.size[0] < const_minSize:
                        a = img_8.size[0] / const_minSize
                        img_8 = img.resize((int(img_8.size[0] // a), int(img_8.size[1] //a)),Image.LANCZOS)
                    if img_8.size[1] < const_minSize:
                        a = img_8.size[1] / const_minSize
                        img_8 = img.resize((int(img_8.size[0] // a), int(img_8.size[1] //a)),Image.LANCZOS)



                    img_8 = self.normalize(self.transform(img_8)) # (3, h, w)

                    self.all_imgs += [self.normalize(img_8)]
                    self.all_imgs_wh += [torch.Tensor([img_w, img_h]).unsqueeze(0)]
                    img_rs = img_rs.view(3, -1).permute(1, 0) # (h*w, 3) RGB
                    self.all_rgbs += [img_rs]
                    self.all_semantics_gt += [semantics_gt]

                    directions = get_ray_directions(img_h, img_w, self.Ks[id_])
                    rays_o, rays_d = get_rays(directions, c2w)
                    rays_t = id_ * torch.ones(len(rays_o), 1)

                    self.all_rays += [torch.cat([rays_o, rays_d,
                                                self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                                                self.fars[id_]*torch.ones_like(rays_o[:, :1]),
                                                rays_t],
                                                1)] # (h*w, 8)
                    
                self.all_rays = torch.cat(self.all_rays, 0) # ((N_images-1)*h*w, 8)
                self.all_rgbs = torch.cat(self.all_rgbs, 0) # ((N_images-1)*h*w, 3)
                self.all_imgs_wh = torch.cat(self.all_imgs_wh, 0) # ((N_images-1)*h*w, 3)
                self.all_semantics_gt = torch.cat(self.all_semantics_gt, 0) # ((N_images-1)*h*w)

        elif self.split in ['val', 'test_train']: # use the first image as val image (also in train)
            # self.val_id = [self.img_ids_train[0], self.img_ids_train[50], self.img_ids_train[100], self.img_ids_train[200], self.img_ids_train[300], self.img_ids_train[400], self.img_ids_train[500]]
            self.val_id = self.img_ids_train[0]

        else: # for testing, create a parametric rendering path
            # test poses and appearance index are defined in eval.py
            pass

    def define_transforms(self):
        # self.resize = T.Resize([400, 600], Image.BICUBIC)
        self.transform = T.ToTensor()
        self.normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def __len__(self):
        if self.split == 'train':
            self.iterations = len(self.all_rays)//self.batch_size
            return self.iterations
        if self.split == 'test_train':
            return self.N_images_train
        if self.split == 'val':
            return self.val_num
        if self.split == 'test_test':
            return self.N_images_test
            # return len(self.poses_test)
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            np.random.seed(global_val.current_epoch * self.iterations + idx)
            sample_ts = np.random.randint(0, len(self.all_imgs))
            img_w, img_h = self.all_imgs_wh[sample_ts]
            img = self.all_imgs[sample_ts]
            # grid
            w_samples, h_samples = torch.meshgrid([torch.linspace(0, 1-1/img_w, int(sqrt(self.batch_size))), \
                                                    torch.linspace(0 , 1-1/img_h, int(sqrt(self.batch_size)))])
            if self.scale_anneal > 0:
                min_scale_cur = min(max(self.min_scale, 1. * exp(-(global_val.current_epoch * self.iterations + idx)* self.scale_anneal)), 0.9)
            else:
                min_scale_cur = self.min_scale
            scale = torch.Tensor(1).uniform_(min_scale_cur, 1.)
            h_offset = torch.Tensor(1).uniform_(0, (1-scale.item())*(1-1/img_h))
            w_offset = torch.Tensor(1).uniform_(0, (1-scale.item())*(1-1/img_w))
            h_sb = h_samples * scale + h_offset
            w_sb = w_samples * scale + w_offset
            h = (h_sb * img_h).floor()
            w = (w_sb * img_w).floor()
            img_sample_points = (w + h * img_w).permute(1, 0).contiguous().view(-1).long()
            uv_sample = torch.cat((h_sb.permute(1, 0).contiguous().view(-1,1), w_sb.permute(1, 0).contiguous().view(-1,1)), -1)

            rgb_sample_points = (img_sample_points + (self.all_imgs_wh[:sample_ts, 0] * self.all_imgs_wh[:sample_ts, 1]).sum()).long()

            sample = {'rays': self.all_rays[rgb_sample_points, :8],
                      'ts': self.all_rays[rgb_sample_points, 8].long(),
                      'rgbs': self.all_rgbs[rgb_sample_points],
                      'semantics_gt': self.all_semantics_gt[rgb_sample_points],
                      'whole_img': img,
                      'rgb_idx': img_sample_points,
                      'min_scale_cur': min_scale_cur,
                      'img_wh': self.all_imgs_wh[sample_ts],
                      'uv_sample': uv_sample}

        elif self.split in ['val', 'test_train', 'test_test']:
            sample = {}
            if self.split == 'val':
                id_ = self.val_id
            elif self.split == 'test_test':
                id_ = self.img_ids_test[idx]
            elif self.split == 'test_train':
                id_ = self.img_ids_train[idx]

            sample['c2w'] = c2w = torch.FloatTensor(self.poses_dict[id_])


            img = Image.open(os.path.join(self.root_dir, 'dense/images',
                                                self.image_paths[id_])).convert('RGB')
            img_w, img_h = img.size
            if self.img_downscale >= 1:
                img_w = img_w//self.img_downscale
                img_h = img_h//self.img_downscale
                img_s = img.resize((img_w, img_h), Image.LANCZOS)


            semantics_gt = torch.Tensor([np.zeros(img_s.size)]).squeeze(dim=0)

            if self.semantics_dir != []:
                path_semantics = os.path.join(self.semantics_dir, self.image_paths[id_].split('.')[0]) + '.pickle'

                print('path semantics:')
                print(path_semantics)

                if os.path.exists(path_semantics):
                    try:
                        with open(path_semantics, 'rb') as f:
                            semantics_gt = torch.Tensor(torch.load(f))
                    except:
                        with open(path_semantics, 'rb') as f:
                            semantics_gt = torch.Tensor(pickle.load(f))

                        if self.use_semantic_function != '':
                            if self.use_semantic_function == 'sigmoid':
                                semantics_gt = torch.sigmoid(semantics_gt)
                            elif self.use_semantic_function == 'double':
                                semantics_gt = semantics_gt ** 2
                            elif self.use_semantic_function == 'triple':
                                semantics_gt = semantics_gt ** 3
                            else:
                                ValueError('no semantic function')

                    semantics_gt = torch.nn.functional.interpolate(semantics_gt.unsqueeze(dim=0).unsqueeze(dim=0), size=(img_h, img_w))

                    semantics_gt = semantics_gt.squeeze(dim=0).permute(1,2,0)

                    semantics_gt[semantics_gt < self.threshold] = 0.01
                    semantics_gt[semantics_gt >= self.threshold] = 1
                else:
                    print('There is no: path_semantics')

            semantics_gt = semantics_gt.reshape(-1,)
            img_w, img_h = img_s.size
            img_s = self.transform(img_s)  # (3, h, w)


            # sample['all_img'] = torch.nn.functional.interpolate(img_s.unsqueeze(dim=0), size=(250, 250)).squeeze(dim=0)
            sample['all_img'] = img_s

            img_s = img_s.view(3, -1).permute(1, 0) # (h*w, 3) RGB
            sample['rgbs'] = img_s

            sample['semantics_gt'] = semantics_gt

            directions = get_ray_directions(img_h, img_w, self.Ks[id_])
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d,
                              self.nears[id_]*torch.ones_like(rays_o[:, :1]),
                              self.fars[id_]*torch.ones_like(rays_o[:, :1])],
                              1) # (h*w, 8)
            sample['rays'] = rays
            sample['ts'] = id_ * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([img_w, img_h])


            sample['rgb_idx'] = torch.LongTensor([i for i in range (0, (img_w*img_h))])
            w_samples, h_samples = torch.meshgrid([torch.linspace(0, 1-1/img_w, int(img_w)), \
                                                    torch.linspace(0, 1-1/img_h, int(img_h))])
            uv_sample = torch.cat((h_samples.permute(1, 0).contiguous().view(-1,1), w_samples.permute(1, 0).contiguous().view(-1,1)), -1)
            sample['uv_sample'] = uv_sample

            img_8 = img.resize((img_w//self.img_downscale_appearance, img_h//self.img_downscale_appearance), Image.LANCZOS)

            const_minSize = 33
            if img_8.size[0] < const_minSize:
                a = img_8.size[0] / const_minSize
                img_8 = img.resize((int(img_8.size[0] // a), int(img_8.size[1] // a)), Image.LANCZOS)
            if img_8.size[1] < const_minSize:
                a = img_8.size[1] / const_minSize
                img_8 = img.resize((int(img_8.size[0] // a), int(img_8.size[1] // a)), Image.LANCZOS)

            img_8 = self.normalize(self.transform(img_8)) # (3, h, w)


            sample['whole_img'] = img_8

        else:
            sample = {}
            sample['c2w'] = c2w = torch.FloatTensor(self.poses_test[idx])
            directions = get_ray_directions(self.test_img_h, self.test_img_w, self.test_K)
            rays_o, rays_d = get_rays(directions, c2w)
            near, far = 0, 5
            rays = torch.cat([rays_o, rays_d,
                              near*torch.ones_like(rays_o[:, :1]),
                              far*torch.ones_like(rays_o[:, :1])],
                              1)
            sample['rays'] = rays
            sample['ts'] = self.test_appearance_idx * torch.ones(len(rays), dtype=torch.long)
            sample['img_wh'] = torch.LongTensor([self.test_img_w, self.test_img_h])

        return sample
