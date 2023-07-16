# LLM-Based Concept Distillation

Download the WikiScenes-based `metadata.csv` file and run:

```
python make_pseudolabels.py -i data/metadata.csv -o data/pseudolabels.csv
```

# Semantic Adaptation

## Step 1: Fine-tune CLIP

Make sure that WikiScenes(+mosques) images are stored under `data/wikiscenes` (or pass another directory to `-d`) and run:

```
python finetune_clip.py -p data/pseudolabels.csv
```

## Step 2: Fine-tune 2D segmentation (CLIPSeg)

TBD