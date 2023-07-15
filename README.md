# HaLo-NeRF: Text-Driven Neural 3D Localization in the Wild

### Coming soon
links to the project page, paper, arxiv, supplementary


Chen Dudai¹, 
Morris Alper¹, 
Hana Bezalel¹, 
Rana Hanocka², 
Itai Lang²,
Hadar Averbuch-Elor¹. 

¹Tel Aviv University,
²The University of Chicago.


This repository is an official implementation of [HaLo-NeRF](https://github.com/TAU-VAILab/HaLo-NeRF/) (Text-Driven Neural 3D Localization in the Wild) using pytorch ([pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning)). 

<!-- The code is largely based on Ha-NeRFNeRF implementation. -->

# Installation

## Hardware

* OS: Ubuntu 20.04
* NVIDIA GPU with CUDA=11.4 (tested with 1 RTXA5000)

## Software

* Clone this repo by `git clone https://github.com/TAU-VAILab/HaLo-NeRF`
* Python>=3.8 (installation via [anaconda](https://www.anaconda.com/distribution/) is recommended, use `conda create -n HaLo-NeRF python=3.8` to create a conda environment and activate it by `conda activate HaLo-NeRF`)
* Python libraries
    * Install core requirements by `pip install -r requirements.txt`
    * Install the following torch packages using the command:
  
      `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`




# Part A: Training The Finetuned Clipseg Model
TODO: Fill here

###  coming soon:
link for downloading the finetuned clipseg model.



# Part B: Training The HaLo-NeRF Model

## Data download - RGB Images and Colmap Model
We are using 6 different scenes:

3 Cathedrals - Milano, St Paul's Cathedral, Notre-Dame

2 Mosques - Badshahi Mosque, Blue-Mosque

1 Synagogue - Hurba 


###  coming soon:
link for downloading the data of the scenes.


## Training the RGB model

Train the RGB model before train the semantic part.

Run:
```
python train_mask_grid_sample.py \
  --root_dir {path to the dataset} --dataset_name phototourism \
  --save_dir save \
  --img_downscale 2 \
  --N_importance 64 --N_samples 64 \
  --num_epochs 20 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name {folder to save} \
  --N_emb_xyz 15 --N_vocab 1500 \
  --encode_a --N_a 48 --weightKL 1e-5 --encode_random --weightRecA 1e-3 --weightMS 1e-6 \
  --num_gpus 1
```

for example:
```
python train_mask_grid_sample.py \
  --root_dir data/st_paul --dataset_name phototourism \
  --save_dir save \
  --img_downscale 2 \
  --N_importance 64 --N_samples 64 \
  --num_epochs 20 --batch_size 1024 \
  --optimizer adam --lr 5e-4 --lr_scheduler cosine \
  --exp_name st_paul \
  --N_emb_xyz 15 --N_vocab 1500 \
  --encode_a --N_a 48 --weightKL 1e-5 --encode_random --weightRecA 1e-3 --weightMS 1e-6 \
  --num_gpus 1
```


`--N_vocab` should be set to an integer larger than the number of images (dependent on different scenes). For example, "notre_dame_front_facade" has in total 3764 images (under `dense/images/`), so any number larger than 3764 works (no need to set to exactly the same number). **Attention!** If you forget to set this number, or it is set smaller than the number of images, the program will yield `RuntimeError: CUDA error: device-side assert triggered` (which comes from `torch.nn.Embedding`).

See [Cfg_file.py](Cfg_file.py) for all configurations.

The checkpoints and logs will be saved to `{save_dir}/ckpts/{scene_name} ` and `{save_dir}/logs/{scene_name}`, respectively.

You can monitor the training process by `tensorboard --logdir {save_dir}/logs/{scene_name} --port=8600` and go to `localhost:8600` in your browser.

###  coming soon:
link for downloading the trained RGB models of the scenes.

# Part C: Training The HaLo-NeRF semantic 

## Retrive the relevant images
TODO: Fill here
###  coming soon:
link for csv retrieval files.

## Create the data for the semantic part

To create semantic data, for training the semantic part - run:
```
python clipseg_ft_horiz_slider.py \
--prompts '{prompt1};{prompt2};{prompt3} etc...' \
--model_path '{the path to the clipseg ft model}' \
--folder_to_save '{folder_to_save}' \
--building_type '{cathedral or mosque or synagogue}' \
--images_folder '{RGB images folder}' \
--csv_retrieval_path '{the path of the retrieval csv file}' \
--n_files {number of files for retrieval}
```

for example:
```
python clipseg_ft_horiz_slider.py \
--prompts "windows;poles;colonnade" \
--model_path data/clipseg_ft_model \
--folder_to_save data/clipseg_ft_inference/st_paul \
--building_type cathedral \
--images_folder data/st_paul/dense/images \
--csv_retrieval_path data/retrieval/st_paul_geometric_occlusions.csv \
--n_files 150
```



You can use as many prompts as you like with this format:
`--prompts '{first prompt};{second prompt};{third prompt}` etc.
for example: 'towers;windows;portals'

###  coming soon:
link for downloading the semantic data of the scenes.

## Train the semantic part

Run:
```
python HaLo-NeRF_pipline.py \
--root_dir {path to the dataset} \
 --xls_path {path to the retrival file for each image} \
 --save_dir ./sem_results/{folder to save} \
 --exp_name test --top_k_files 150 --num_epochs 10 \
 --ckpt_path {ckpt path} \
 --N_vocab 1500 --prompts '{first prompt};{second prompt};{third prompt} etc...' --scene_name {scene name} \
 --train_HaloNeRF_flag \
 --semantics_dir {path for semantic data}
 --max_steps 12500
```

For example:
```
python HaLo-NeRF_pipline.py \
--root_dir data/st_paul \
 --xls_path data/retrieval/st_paul_geometric_occlusions.csv \
 --save_dir ./sem_results/st_paul_save \
 --exp_name test --top_k_files 150 --num_epochs 10 \
 --ckpt_path ./save/ckpts/st_paul/epoch=19.ckpt \
 --N_vocab 1500 --prompts "portals;towers;windows" --scene_name st_paul \
 --train_HaloNeRF_flag \
  --semantics_dir data/clipseg_ft_inference/st_paul/clipseg_ft/
  --max_steps 12500
 ```

If you want to calculate the metrics please add the following flags:
```
--save_for_metric_flag 
--calc_metrics_flag
--path_gt {/path/to/the/ground_truth_masks} (for example: data/manually_gt_masks_0_1/)
```

Please notice that the "max_steps" flag defines the maximum number of iterations for training the semantic part.

### coming soon:
link for downloading the trained models of the scenes with the semantic part.

also, we will add the link for the semantic ground truth masks.



# Part D: Evaluation

Use [eval.py](eval.py) to inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered
images.

Run:
```
python eval.py \
  --root_dir {path to the dataset} \
  --save_dir save \
  --dataset_name phototourism --scene_name {scene name} \
  --split {test / test_train / val} --img_downscale 2 \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500 --encode_a \
  --ckpt_path {the path of the CKPT of the model} \
  --chunk 16384 --img_wh {image size} \
  --enable_semantic \
  --save_imgs
```

For example:
```
python eval.py \
  --root_dir data/st_paul \
  --save_dir save \
  --dataset_name phototourism --scene_name st_paul \
  --split test_train --img_downscale 2 \
  --N_samples 256 --N_importance 256 --N_emb_xyz 15 \
  --N_vocab 1500 --encode_a \
  --ckpt_path  ./sem_results/st_paul_save/ckpts/test/windows/epoch=3.ckpt \
  --chunk 16384 --img_wh 320 240 \
  --enable_semantic \
  --save_imgs
```

The 'split' field defines on which dataset to run the evaluation:

    test_train - will run it on the train dataset.
    val - will run it on the validation dataset.
    test - will run it on your chosen cameras locations and interpolate between them.

Please notice that if you use 'test' as split you can use the following flags:
```
--num_frames {number of frames to interpolate}
--images_ids {images IDs locations to interpolate}
```
for example:
```
--num_frames [24, 8]
--images_ids [40, 588]
```



# Cite
If you find our work useful, please consider citing:
### Coming soon - cite
```
```

# Acknowledge
Our code is based on the pytorch implementation of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/). We appreciate all the contributors.
