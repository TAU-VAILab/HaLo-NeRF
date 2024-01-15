# 3D Localization

## Step 1: Train The 3D RGB Model

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

## Step 2: Create the RGB images from the NeRF

If you want to use the occlusion section in the retrieval you will need the RGB results from the NeRF model.
You can skip this part and not use the `--use_occlusion` flag in step 3, or if you are using `--ignore_csv_use_all_images` flag in step 5.

For creating the RGB results please run:
```
python eval.py --root_dir data/{scene name}/ --save_dir save \
--dataset_name phototourism --scene_name {folder name to save} \
--split train --img_downscale 2 --N_samples 64 --N_importance 64 \
--N_emb_xyz 15 --N_vocab 4000 --encode_a \
--ckpt_path {ckpt path} --chunk 16384
```

for example 
```
Python eval.py --root_dir data/st_paul/ --save_dir save \
--dataset_name phototourism --scene_name st_paul_rgb \
--split train --img_downscale 2 --N_samples 64 --N_importance 64 \
--N_emb_xyz 15 --N_vocab 4000 --encode_a \
--ckpt_path ./save/ckpts/st_paul/epoch=19.ckpt --chunk 16384
```


## Step 3: Retrieve relevant images

```
python run_retrieval.py -b {building type} \
--images_folder '{RGB images folder}' \
--rgb_reconstruction_folder '{RGB reconstruction images folder}' \
-o {output filename}
--use_occlusion
```

If you are not using the occlusion please remove the `--use_occlusion` flag.


The building type may be e.g. "cathedral", "mosque", "synagogue", "all" (for not using a building type) etc.

For example:

```
python run_retrieval.py -b cathedral \
--images_folder data/st_paul/dense/images \
--rgb_reconstruction_folder data/nerf/st_paul \
-o data/retrieval/st_paul_geometric_occlusions.csv
```

Note: The filenames of corresponding RGB images and RGB reconstructions must match up to leading zeros (e.g. `100.jpg` vs. `0100.jpg`) and are assumed to be up to four digits long.

## Step 4: Perform 2D segmentation on images

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

Please notice that you can create the semantic data of all the images without using `csv_retrieval_path` by setting the flag `use_csv_for_retrieval` to False. 

## Step 5: Train the 3D semantic model on 2D segmentations

Run:
```
python train_semantic.py \
--root_dir {path to the dataset} \
 --csv_retrieval_path {path to the retrival file for each image} \
 --save_dir ./sem_results/{folder to save} \
 --exp_name test --top_k_files 150 --num_epochs 10 \
 --ckpt_path {ckpt path} \
 --N_vocab 1500 --prompts '{first prompt};{second prompt};{third prompt} etc...' --scene_name {scene name} \
 --train_HaloNeRF_flag \
 --semantics_dir {path for semantic data} \
 --max_steps 12500
```

For example:
```
python train_semantic.py \
--root_dir data/st_paul \
 --csv_retrieval_path data/retrieval/st_paul_geometric_occlusions.csv \
 --save_dir ./sem_results/st_paul_save \
 --exp_name test --top_k_files 150 --num_epochs 10 \
 --ckpt_path ./save/ckpts/st_paul/model.ckpt \
 --N_vocab 1500 --prompts "portals;towers;windows" --scene_name st_paul \
 --train_HaloNeRF_flag \
  --semantics_dir data/clipseg_ft_inference/st_paul/clipseg_ft/ \
  --max_steps 12500
 ```

If you want to calculate the metrics please add the following flags:
```
--save_for_metric_flag 
--calc_metrics_flag
--path_gt {/path/to/the/ground_truth_masks} (for example: data/HolyScenes/cathedral/st_paul/)
```

Please notice that the "max_steps" flag defines the maximum number of iterations for training the semantic model.

Please notice also that you can run step 5 on all the images (and skip step 2 and step 3) by using the flag `--ignore_csv_use_all_images`.
