# HaLo-NeRF: Text-Driven Neural 3D Localization in the Wild

### TODO: change

**[Project Page](https://rover-xingyu.github.io/Ha-NeRF/) |
[Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Hallucinated_Neural_Radiance_Fields_in_the_Wild_CVPR_2022_paper.pdf) |
[Latest arXiv](https://arxiv.org/pdf/2111.15246.pdf) |
[Supplementary](https://rover-xingyu.github.io/Ha-NeRF/files/Ha_NeRF_CVPR_2022_supp.pdf)**

[Xingyu Chen¹](https://rover-xingyu.github.io/), 
[Qi Zhang²](https://qzhang-cv.github.io/), 
[Xiaoyu Li²](https://xiaoyu258.github.io/), 
[Yue Chen¹](https://fanegg.github.io/), 
[Ying Feng²](https://github.com/rover-xingyu/Ha-NeRF/),
[Xuan Wang²](https://scholar.google.com/citations?user=h-3xd3EAAAAJ&hl=en/),
[Jue Wang²](https://juewang725.github.io/). 

[¹Xi'an Jiaotong University)](http://en.xjtu.edu.cn/),
[²Tencent AI Lab](https://ai.tencent.com/ailab/en/index/).


This repository is an official implementation of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/) (Hallucinated Neural Radiance Fields in the Wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). 

<!-- I try to reproduce (some of) the results on the lego dataset (Section D). Training on [Phototourism real images](https://github.com/ubc-vision/image-matching-benchmark) (as the main content of the paper) has also passed. Please read the following sections for the results.

The code is largely based on NeRF implementation (see master or dev branch), the main difference is the model structure and the rendering process, which can be found in the two files under `models/`. -->

# :computer: Installation

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with CUDA=11.4 (tested with 1 RTXA5000)

## Software

* Clone this repo by `git clone https://github.com/chendudai/Ha-NeRF`
* Python>=3.6 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n HaNeRF python=3.6` to create a conda environment and activate it by `conda activate HaNeRF`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`

###  TODO: remove after

you can use instead:  `conda create --name myclone --clone /storage/chendudai/envs/conda/envs/Ha-NeRF4/`
    

# :key: Training

## Data download

###  TODO: change after

Download the scenes you want from [here](https://www.cs.ubc.ca/~kmyi/imw2020/data.html).

The link is not working. please take the scenes from here: `/storage/chendudai/data/`



## Training the RGB model

### TODO: remove after
The trained RGB models can be found here:
`/storage/chendudai/repos/Ha-NeRF/save/ckpts/`

Train the RGB model before train the semantic part.

Run (example)
```
python train_mask_grid_sample.py \
  --root_dir /path/to/the/dataset/ --dataset_name phototourism \
  --save_dir save \
  --img_downscale 2 \
  --N_importance 64 --N_samples 64 \
  --num_epochs 20 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name exp_HaNeRF_Brandenburg_Gate \
  --N_emb_xyz 15 --N_vocab 1500 \
  --encode_a --N_a 48 --weightKL 1e-5 --encode_random --weightRecA 1e-3 --weightMS 1e-6 \
  --num_gpus 1
```

`--N_vocab` should be set to an integer larger than the number of images (dependent on different scenes). For example, "notre_dame_front_facade" has in total 3764 images (under `dense/images/`), so any number larger than 3764 works (no need to set to exactly the same number). **Attention!** If you forget to set this number, or it is set smaller than the number of images, the program will yield `RuntimeError: CUDA error: device-side assert triggered` (which comes from `torch.nn.Embedding`).

See [Cfg_file.py](Cfg_file.py) for all configurations.

The checkpoints and logs will be saved to `{save_dir}/ckpts/{scene_name} ` and `{save_dir}/logs/{scene_name}`, respectively.

You can monitor the training process by `tensorboard --logdir {save_dir}/logs/{scene_name} --port=8600` and go to `localhost:8600` in your browser.

# Create the data for the semantic part

You can use the created semantic data, or create semantic data of your own.
the created data is saved here:
`/storage/chendudai/data/clipseg_ft_crops_refined_plur_newcrops_10epochs/`

To create semantic data of your own, run:
```
python clipsef_ft_horiz_slider.py \
--prompts '{prompt1};{prompt2};{prompt3}' \
--scene_name '{scene_name}' \
--folder_to_save '{folder_to_save}' \
--building_type '{cathedral or mosque or synagogue}' \
--data_folder '{data_folder}'
```

for example:
```
python clipsef_ft_horiz_slider.py \
--prompts 'spires;poles;colonnade' \
--scene_name 'st_paul' \
--folder_to_save 'st_paul/horizontal' \
--building_type 'cathedral' \
--data_folder '/storage/chendudai/data/st_paul'
```


# Train the semantic part

Use [eval.py](eval.py) to inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered
images.

Run (example)
```
python SeRF_pipline.py \
--root_dir {/path/to/the/dataset} \
 --xls_path {/path/to/the/retrival file for each image} \
 --save_dir ./sem_results/{folder to save} \
 --exp_name test --top_k_files 150 --num_epochs 10 \
 --ckpt_path ./save/ckpts/0_1_withoutSemantics/epoch=15.ckpt \
 --N_vocab 1500 --prompts '{first prompt};{second prompt};{third prompt}' --scene_name {scene name} \
 --train_SeRF_flag
```

You can use as many prompts as you like with this format:
`--prompts '{first prompt};{second prompt};{third prompt}` etc.
for example: 'towers;windows;portals'

`--xls_path` is the path of the file that contains the retrival score for each image.
for example:
/storage/chendudai/data/notre_dame_geometric_occlusions.csv

If you want to calculate the metrics please add the following flags:
`--save_for_metric_flag
--calc_metrics_flag 
--path_gt /path/to/the/ground_truth_masks
`
### TODO: remove after
You can use the path_gt for example:
`--path_gt /storage/chendudai/data/gt_warps_with_manually_100/{scene_name}`
# Evaluation

Use [eval.py](eval.py) to inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered
images.

Run (example)
```
python eval.py \
  --root_dir /path/to/the/datasets/brandenburg_gate/ \
  --save_dir save \
  --dataset_name phototourism --scene_name HaNeRF_Trevi_Fountain \
  --split test_test --img_downscale 2 \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500 --encode_a \
  --ckpt_path save/ckpts/HaNeRF_Brandenburg_Gate/epoch\=19.ckpt \
  --chunk 16384 --img_wh 320 240
```


# TODO: change cite

# Cite
If you find our work useful, please consider citing:
```bibtex
@inproceedings{chen2022hallucinated,
  title={Hallucinated neural radiance fields in the wild},
  author={Chen, Xingyu and Zhang, Qi and Li, Xiaoyu and Chen, Yue and Feng, Ying and Wang, Xuan and Wang, Jue},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12943--12952},
  year={2022}
}
```

# Acknowledge
Our code is based on the awesome pytorch implementation of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/). We appreciate all the contributors.
