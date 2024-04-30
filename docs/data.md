# Data and Downloads

Download files for all provided data and pretrained models and results are listed below.
All the files are in the following folder: [Drive Folder](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/Eos1ebfo2hVMhAVsGKAp9MEBGrqIro7LDIZ56rVt5FqHMw?e=h9LWot)
## Data for finetuning the 2D segmentation model

This data uses for fine-tuning the 2D segmentation model CLIPSeg using the Wikiscenes dataset. (more about it here: [distillation_adaptation.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/346974a807861516822240ad665807ae46c28755/docs/distillation_adaptation.md))

* WikiScenes (cathedrals+mosques) image data:
  * Available from [WikiScenes repo](https://github.com/tgxs002/wikiscenes/tree/main) (high-res 1200px images)
  * Store these in `data/wikiscenes/cathedrals` and `data/wikiscenes/mosques`
* WikiScenes (+mosques) metadata: `metadata.csv`
* Distilled semantic pseudo-labels `pseudolabels.csv` for WikiScenes (+mosques)
* Metadata files: `seg_crop_metadata.csv`, `seg_hash_metadata.csv`.  extract to `data/` folder.
* Correspondence-based data: `hashdata.tar.gz`; extract to `data/hashdata`

## Test scene data

* Test scene images and COLMAP reconstructions: `rgb_data.tar.gz`.  
  (This dataset uses for training the HaLo-NeRF, more about it here: [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)). 
* GT semantic masks: `HolyScenes.tar.gz` (It uses for evaluating the 2D segmentation model and 3D localization).
* RGB reconstructions: `rgb_reconstructions.tar.gz` (It uses for retrieval (in `run_retrieval.py`, more about it here: [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)).

## Pretrained models

* Fine-tuned CLIP model `clip_ft.tar.gz` (It uses for fine-tuning the 2D segmentation model here: `finetune_seg.py` [distillation_adaptation.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/346974a807861516822240ad665807ae46c28755/docs/distillation_adaptation.md))
* Fine-tuned Segmentation model: `clipseg_ft.tar.gz` (It uses for the 3D localization. More about it here: [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)).  
* Trained RGB NeRF models: `rgb_models.tar.gz` (The trained RGB NeRF models after the `train_rgb.py` part in here [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)).
* Trained Semantic NeRF models: `semantic_models.tar.gz` (The trained RGB NeRF models after the `train_rgb.py` part in here [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)).

## Inference results

* Retrieval data: `retrieval.tar.gz` (The results of the retrieval part `run_retrieval.py` in here:  [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)).
* Scene semantic segmentation data: `semantic_data.tar.gz` (The results of the 2D semantic segmentation part `run_segmentation.py`  in here:  [3d_localization.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/3d_localization.md)).
* RGB NeRF results: `nerf_rgb.tar.gz` (The results of the RGB prediction of the HaLo-NeRF: using the `eval.py` code. See more here: [evaluation.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/evaluation.md)).
* Semantic HaLo-NeRF results: `semantic_results.tar.gz` (The results of the semantic prediction of the HaLo-NeRF: using the `eval.py` code. See more here: [evaluation.md](https://github.com/TAU-VAILab/HaLo-NeRF/blob/main/docs/evaluation.md)).
