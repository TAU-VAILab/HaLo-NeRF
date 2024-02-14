# Data and Downloads

Download links for all provided data and pretrained models are listed below.
All the links are in the folder: [Drive Folder](https://tauex-my.sharepoint.com/:f:/g/personal/hadarelor_tauex_tau_ac_il/Eos1ebfo2hVMhAVsGKAp9MEBGrqIro7LDIZ56rVt5FqHMw?e=h9LWot)
## WikiScenes (+mosques)-based data

* WikiScenes (+mosques) image data: (link TBA)
  * Partially available from [WikiScenes repo](https://github.com/tgxs002/wikiscenes/tree/main) (high-res 1200px images)
  * Store these in `data/wikiscenes/cathedrals` and `data/wikiscenes/mosques`
* WikiScenes (+mosques) metadata (`metadata.csv`) ([link](https://tauex-my.sharepoint.com/:x:/r/personal/hadarelor_tauex_tau_ac_il/_layouts/15/Doc.aspx?sourcedoc=%7BA75B02C1-6BD5-4D28-A9DE-EA282A86DCAB%7D&file=metadata.csv&action=default&mobileredirect=true))
* Distilled semantic pseudo-labels (`pseudolabels.csv`) for WikiScenes (+mosques) ([link](https://tauex-my.sharepoint.com/:x:/r/personal/hadarelor_tauex_tau_ac_il/_layouts/15/Doc.aspx?sourcedoc=%7B1DFED3BF-3C7C-4DE5-8F79-CFF4DCC50B25%7D&file=pseudolabels.csv&action=default&mobileredirect=true))
* Data for finetuning segmentation:
  * Metadata files: `data/seg_crop_metadata.csv` ([link](https://tauex-my.sharepoint.com/:x:/r/personal/hadarelor_tauex_tau_ac_il/_layouts/15/Doc.aspx?sourcedoc=%7B9DD23269-89F2-4531-8C8B-7BB3FF2B1569%7D&file=seg_crop_metadata.csv&action=default&mobileredirect=true)), `data/seg_hash_metadata.csv` ([link](https://tauex-my.sharepoint.com/:x:/r/personal/hadarelor_tauex_tau_ac_il/_layouts/15/Doc.aspx?sourcedoc=%7B7ABC9B99-B333-43D1-9BE4-1A33E7BCB685%7D&file=seg_hash_metadata.csv&action=default&mobileredirect=true)) 
  * Correspondence-based data: `hashdata.tar.gz` ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fhashdata%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF)); extract to `data/hashdata`

## Test scene data

* Test scene images and COLMAP reconstructions: (`rgb_data.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Frgb%5Fdata%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* GT semantic masks (`HolyScenes.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2FHolyScenes%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* RGB reconstructions (`rgb_reconstructions.tar.gz`, used for retrieval): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Frgb%5Freconstructions%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))

## Pretrained models

* Fine-tuned CLIP model (`clip_ft.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fclip%5Fft%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* Segmentation model (`clipseg_ft.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fclipseg%5Fft%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* RGB NeRF models: (`rgb_models.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Frgb%5Fmodels%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* Semantic NeRF models: (`semantic_models.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fsemantic%5Fmodels%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))

## Inference results

* Retrieval data: (`retrieval.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fretrieval%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* Scene semantic segmentation data: (`semantic_data.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fsemantic%5Fdata%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* RGB NeRF results: (`nerf_rgb.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fnerf%5Frgb%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
* Semantic HaLo-NeRF results: (`semantic_results.tar.gz`): ([link](https://tauex-my.sharepoint.com/personal/hadarelor_tauex_tau_ac_il/_layouts/15/onedrive.aspx?ga=1&id=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF%2Fsemantic%5Fresults%2Etar%2Egz&parent=%2Fpersonal%2Fhadarelor%5Ftauex%5Ftau%5Fac%5Fil%2FDocuments%2FHaLo%2DNeRF))
