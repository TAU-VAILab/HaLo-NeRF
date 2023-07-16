# LLM-Based Concept Distillation

```
python make_pseudolabels.py -i data/metadata.csv -o data/pseudolabels.csv
```

# Semantic Adaptation

## Step 1: Fine-tune CLIP

```
python finetune_clip.py -p data/pseudolabels.csv
```