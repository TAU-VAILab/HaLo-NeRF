# Data and Downloads

Download links for all provided data and pretrained models are listed below.

## WikiScenes (+mosques)-based data

* WikiScenes (+mosques) image data: (link TBA)
  * Partially available from [WikiScenes repo](https://github.com/tgxs002/wikiscenes/tree/main) (high-res 1200px images)
  * Store these in `data/wikiscenes/cathedrals` and `data/wikiscenes/mosques`
* WikiScenes (+mosques) metadata (`metadata.csv`) ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))
* Distilled semantic pseudo-labels (`pseudolabels.csv`) for WikiScenes (+mosques) ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))
* Data for finetuning segmentation:
  * Metadata files: `data/seg_crop_metadata.csv`, `data/seg_hash_metadata.csv` ([link for both](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))
  * Correspondence-based data: `hashdata.tar.gz`; extract to `data/hashdata` ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))

## Test scene data

* Test scene images and COLMAP reconstructions: (`rgb_data.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))
* GT semantic masks (`HolyScenes.tar.gz`): ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))
* RGB reconstructions (`rgb_reconstructions.tar.gz`, used for retrieval): ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))

## Pretrained models

* Fine-tuned CLIP model (`clip_ft.tar.gz`): ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))
* Segmentation model (`clipseg_ft.tar.gz`): ([link](https://drive.google.com/drive/folders/1n-MH0MPBQ-efxFNQPAqh4TwoMQcoU4AD?usp=sharing))
* RGB NeRF models: (`rgb_models.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))
* Semantic NeRF models: (`semantic_models.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))

## Inference results

* Retrieval data: (`retrieval.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))
* Scene semantic segmentation data: (`semantic_data.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))
* RGB NeRF results: (`nerf_rgb.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))
* Semantic HaLo-NeRF results: (`semantic_results.tar.gz`): ([link](https://drive.google.com/drive/u/1/folders/1I-sexE7wJTK3YJjI1F_TdR0Vn0krYe73))