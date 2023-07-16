# HaLo-NeRF: Text-Driven Neural 3D Localization in the Wild

(TBA) **project page | paper**

*Chen Dudai¹, 
[Morris Alper](https://morrisalp.github.io/)¹, 
Hana Bezalel¹, 
[Rana Hanocka](https://people.cs.uchicago.edu/~ranahanocka/)², 
[Itai Lang](https://scholar.google.com/citations?user=q0bBhtsAAAAJ)²,
[Hadar Averbuch-Elor](https://www.elor.sites.tau.ac.il/)¹*

*¹[Tel Aviv University](https://english.tau.ac.il/),
²[The University of Chicago](https://www.uchicago.edu/)*


This repository is the official implementation of [HaLo-NeRF](https://github.com/TAU-VAILab/HaLo-NeRF/) (Text-Driven Neural 3D Localization in the Wild).

# Requirements and Installation

## Hardware

Tested on:
* OS: Ubuntu 20.04
* NVIDIA GPU with CUDA=11.4 (tested with 1 RTXA5000)

## Installation Instructions

* Clone this repo with `git clone https://github.com/TAU-VAILab/HaLo-NeRF`
* Run in a Python>=3.8 environment. Recommended: create and use conda environment via `conda create -n HaLo-NeRF python=3.8` and `conda activate HaLo-NeRF`
* Install core requirements with `pip install -r requirements.txt`
* Install the following torch packages using the command:
  
      `pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`

# Data Downloads

You may download the relevant scenes and pretrained models at the links below.

Download links:

* Scene images and COLMAP reconstructions: (Link TBA)
* GT semantic masks: (Link TBA)
* Segmentation model: (Link TBA)
* Scene semantic segmentation data: (Link TBA)
* Retrieval data: (Link TBA)
* RGB NeRF models: (Link TBA)
* Semantic NeRF models: (Link TBA)


Note that we use six different scenes:

* Three cathedrals - [Milan Cathedral](https://en.wikipedia.org/wiki/Milan_Cathedral), [St Paul's Cathedral](https://en.wikipedia.org/wiki/St_Paul%27s_Cathedral), [Notre-Dame](https://en.wikipedia.org/wiki/Notre-Dame_de_Paris)

* Two mosques - [Badshahi Mosque](https://en.wikipedia.org/wiki/Badshahi_Mosque), [Blue Mosque](https://en.wikipedia.org/wiki/Blue_Mosque,_Istanbul)

* One synagogue - [Hurva Synagogue](https://en.wikipedia.org/wiki/Hurva_Synagogue) 

# Training

Note: For each command, you may pass `--help` to see additional flags and configuration options.

## Part A: Training The Finetuned Segmentation Model
TODO: Fill here

## Part B: Training The HaLo-NeRF RGB Model

Train the RGB model before training the semantic model.

Run:
```
python train_rgb.py \
  --root_dir {path to the dataset} --dataset_name phototourism --save_dir save \
  --num_epochs 20 --exp_name {folder to save} --N_vocab 1500
```

for example:
```
python train_rgb.py \
  --root_dir data/st_paul --dataset_name phototourism --save_dir save \
  --num_epochs 20 --exp_name st_paul --N_vocab 1500
```

`--N_vocab` should be set to an integer larger than the number of images (dependent on different scenes). For example, "notre_dame_front_facade" has in total 3764 images (under `dense/images/`), so any number larger than 3764 works (no need to set to exactly the same number). **Attention!** If you forget to set this number, or it is set smaller than the number of images, the program will yield `RuntimeError: CUDA error: device-side assert triggered` (which comes from `torch.nn.Embedding`).

The checkpoints and logs will be saved to `{save_dir}/ckpts/{scene_name} ` and `{save_dir}/logs/{scene_name}`, respectively.

You can monitor the training process by `tensorboard --logdir {save_dir}/logs/{scene_name} --port=8600` and go to `localhost:8600` in your browser.

# Part C: Training The HaLo-NeRF Semantic Model

## Retrive the relevant images
TODO: Fill here

## Create the data for the semantic model

To create semantic data, for training the semantic model - run:
```
python run_segmentation.py \
--prompts '{prompt1};{prompt2};{prompt3} etc...' \
--model_path '{the path to the segmentation model}' \
--folder_to_save '{folder_to_save}' \
--building_type '{cathedral or mosque or synagogue}' \
--images_folder '{RGB images folder}' \
--csv_retrieval_path '{the path of the retrieval csv file}' \
--n_files {number of files for retrieval}
```

for example:
```
python run_segmentation.py \
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

## Train the semantic model

Run:
```
python train_semantic.py \
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
python train_semantic.py \
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

Please notice that the "max_steps" flag defines the maximum number of iterations for training the semantic model.


# Part D: Evaluation

Use [eval.py](eval.py) to inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered
images.

Run:
```
python eval.py \
  --root_dir {path to the dataset} \
  --save_dir save \
  --dataset_name phototourism --scene_name {scene name} \
  --split {test / test_train / val} \
  --N_vocab 1500 \
  --ckpt_path {the path of the CKPT of the model} \
  --img_wh {image size}
```

For example:
```
python eval.py \
  --root_dir data/st_paul \
  --save_dir save \
  --dataset_name phototourism --scene_name st_paul \
  --split test_train \
  --N_vocab 1500 \
  --ckpt_path  ./sem_results/st_paul_save/ckpts/test/windows/epoch=3.ckpt \
  --img_wh 320 240
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



# Citation
If you find this project useful, you may cite us as follows:
```
(TBA)
```

# Acknowledgements
This implementation is based on the official repository of [Ha-NeRF](https://rover-xingyu.github.io/Ha-NeRF/).
