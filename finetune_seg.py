# from glob import glob
# from hashdata import HashData
# from tqdm.auto import tqdm, trange
# from torchinfo import summary
# import torch
# from matplotlib import pyplot as plt
# import cv2
# from torch import nn
# import pandas as pd
# import pickle
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import CLIPProcessor, CLIPModel
# from torch.utils.data import Dataset, DataLoader, IterableDataset
# from pairs import resize_img
# from collections import namedtuple
# import numpy as np
# from torchvision.transforms import Resize
# from PIL import Image
from argparse import ArgumentParser
import os

def get_opts():
    parser = ArgumentParser()
    # parser.add_argument('--pseudolabels', '-p', type=str, required=True, help="filename of pseudolabel csv file")
    # parser.add_argument('--epochs', '-e', type=int, default=5, help="epochs")
    # parser.add_argument('--lr', '-l', type=float, default=1e-6, help="learning rate")
    # parser.add_argument('--batch_size', '-b', type=int, default=2 ** 7, help="batch size")
    # parser.add_argument('--num_workers', '-n', type=int, default=0, help="number of dataloader workers")
    # parser.add_argument('--output', '-o', type=str, default="data/clip_ckpt", help="output checkpoint directory")
    # parser.add_argument('--data_dir', '-d', type=str, default="data/wikiscenes", help="directory data is stored in (under /")

    parser.add_argument('--clip_ft', '-c', type=str, default="data/clip_ckpt/0_CLIPModel", help="directory of fine-tuned CLIP checkpoint")

    return parser.parse_args()

def main():
    args = get_opts()

    print("Loading models...")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to('cuda')

    model_orig = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_orig.to('cuda')
    model_orig.eval()

    for param in model_orig.parameters():
        param.requires_grad = False

    assert os.path.exists(args.clip_ft), f'Missing finetuned CLIP directory: {args.clip_ft}'
    clip_proc = CLIPProcessor.from_pretrained(args.clip_ft)
    clip = CLIPModel.from_pretrained(args.clip_ft)
    clip.to('cuda')
    clip.eval()

    print("Models loaded")

if __name__ == "__main__":
    main()
