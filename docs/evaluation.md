# Evaluation

Use [eval.py](eval.py) to run inference on all test data. It will create folder `{save_dir}/results/{dataset_name}/{scene_name}` and save the rendered
images.

Run:
```
python eval.py \
  --root_dir {path to the dataset} \
  --save_dir save \
  --dataset_name phototourism --scene_name {scene name} \
  --split {test / train} \
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
  --split train \
  --N_vocab 1500 \
  --ckpt_path  ./sem_results/st_paul_save/ckpts/test/windows/epoch=3.ckpt \
  --img_wh 320 240
```

The 'split' field defines on which dataset to run the evaluation:

    train - will run it on the training dataset.
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



# Appearance change
In addition, if you like to change the appearance of the scene you can use [eval_appearance.py](eval_appearance.py).
It will interpolate two different appearances of two images.

Run:
```
python eval_appearance.py \
--root_dir {path to the dataset} \
--save_dir save \
--dataset_name phototourism --scene_name {scene name} \
--split {test / train} \
--N_vocab 1500 \
--ckpt_path {the path of the CKPT of the model} \
--img_wh {image size}
--images_ids {images ids to eval} \ 
--images_id_appearance_first {the appearance of the image ID you want to start with} \
--images_id_appearance_last {the appearance of the image ID you want to end with} \
--num_frames {number of frames}
```

For example:
```
python eval_appearance.py \
--root_dir data/st_paul \
--save_dir save \
--dataset_name phototourism --scene_name st_paul \
--split test \
--N_vocab 1500 \
--ckpt_path  ./sem_results/st_paul_save/ckpts/test/windows/epoch=3.ckpt \
--img_wh 320 240 \
--images_ids 35 35 \ 
--images_id_appearance_first 73 \
--images_id_appearance_last 1 \
--num_frames 48 1
```
