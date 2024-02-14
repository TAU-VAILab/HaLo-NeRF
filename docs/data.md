# Data and Downloads

Download files for all provided data and pretrained models and results are listed below.
All the files are in the following folder: [Drive Folder](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/Eos1ebfo2hVMhAVsGKAp9MEBGrqIro7LDIZ56rVt5FqHMw?e=h9LWot)
## WikiScenes (+mosques)-based data

* WikiScenes (+mosques) image data: (link TBA)
  * Partially available from [WikiScenes repo](https://github.com/tgxs002/wikiscenes/tree/main) (high-res 1200px images)
  * Store these in `data/wikiscenes/cathedrals` and `data/wikiscenes/mosques`
* WikiScenes (+mosques) metadata (`metadata.csv`)
* Distilled semantic pseudo-labels (`pseudolabels.csv`) for WikiScenes (+mosques)
* Data for finetuning segmentation:
  * Metadata files: `seg_crop_metadata.csv`, `seg_hash_metadata.csv`.  extract to `data/` folder.
  * Correspondence-based data: `hashdata.tar.gz`; extract to `data/hashdata`

## Test scene data

* Test scene images and COLMAP reconstructions: (`rgb_data.tar.gz`)
* GT semantic masks (`HolyScenes.tar.gz`)
* RGB reconstructions (`rgb_reconstructions.tar.gz`, used for retrieval):

## Pretrained models

* Fine-tuned CLIP model (`clip_ft.tar.gz`)
* Segmentation model (`clipseg_ft.tar.gz`)
* RGB NeRF models: (`rgb_models.tar.gz`)
* Semantic NeRF models: (`semantic_models.tar.gz`)

## Inference results

* Retrieval data: (`retrieval.tar.gz`)
* Scene semantic segmentation data: (`semantic_data.tar.gz`)
* RGB NeRF results: (`nerf_rgb.tar.gz`)
* Semantic HaLo-NeRF results: (`semantic_results.tar.gz`)
