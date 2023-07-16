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

## Step 2: Retrieve relevant images

TODO

## Step 3: Perform 2D segmentation on images

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

## Step 4: Train the 3D semantic model on 2D segmentations

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
 --semantics_dir {path for semantic data} \
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
  --semantics_dir data/clipseg_ft_inference/st_paul/clipseg_ft/ \
  --max_steps 12500
 ```

If you want to calculate the metrics please add the following flags:
```
--save_for_metric_flag 
--calc_metrics_flag
--path_gt {/path/to/the/ground_truth_masks} (for example: data/manually_gt_masks_0_1/)
```

Please notice that the "max_steps" flag defines the maximum number of iterations for training the semantic model.


