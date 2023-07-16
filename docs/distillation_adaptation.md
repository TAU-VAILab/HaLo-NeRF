# LLM-Based Concept Distillation

Download the WikiScenes-based `metadata.csv` file and run:

```
python make_pseudolabels.py -i data/metadata.csv -o data/pseudolabels.csv
```

# Semantic Adaptation

## Step 1: Fine-tune CLIP

Make sure that WikiScenes(+mosques) images are stored under `data/wikiscenes` (or pass another directory to `-d`) and run:

```
python finetune_clip.py
```

This assumes the pseudolabel data is at `data/pseudolabels.csv`; you can pass a different directory with `-p`.

This uses a single dataloader worker by default; add `-n` with a positive integer to use more workers for possibly faster training.

This saves checkpoints to `data/clip_ckpt` by default.

## Step 2: Fine-tune 2D segmentation (CLIPSeg)

```
python finetune_seg.py
```

This by default looks for the fine-tuned CLIP checkpoint in `data/clip_ckpt/0_CLIPModel`; you may pass a different directory with `-c`.

Note: This requires various data and metadata files as described in the [data docs](data.md). Pass `--help` to see the default assumed locations of these.

