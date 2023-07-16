# from glob import glob
# from hashdata import HashData
# from tqdm.auto import tqdm, trange
# from torchinfo import summary
# import torch
# from matplotlib import pyplot as plt
# import cv2
# from torch import nn
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
import pandas as pd
from seg_utils.hash_data import HashDS
from seg_utils.crop_data import rand_crop, search_crops

def get_opts():
    parser = ArgumentParser()
    
    parser.add_argument('--clip_ft', '-c', type=str, default="data/clip_ckpt/0_CLIPModel", help="directory of fine-tuned CLIP checkpoint")
    parser.add_argument('--data_dir', '-d', type=str, default="data/wikiscenes", help="directory WikiScenes data is stored in")
    parser.add_argument('--crop_metadata', '-cm', type=str, default="data/seg_crop_metadata.csv", help="crop metadata filename")
    parser.add_argument('--hash_metadata', '-hm', type=str, default="data/seg_hash_metadata.csv", help="hash metadata filename")
    parser.add_argument('--hash_data', '-hd', type=str, default="data/hashdata", help="hash data (correspondence-based data) directory")
    parser.add_argument('--pseudolabels', '-p', type=str, default="data/pseudolabels.csv", help="filename of pseudolabel csv file")

    return parser.parse_args()

def main():
    args = get_opts()

    assert os.path.exists(args.crop_metadata), f'Missing crop metadata file: {args.crop_metadata}'
    assert os.path.exists(args.hash_metadata), f'Missing hash metadata file: {args.hash_metadata}'
    assert os.path.exists(args.hash_data), f'Missing hash data directory: {args.hash_metadata}'
    assert os.path.exists(args.clip_ft), f'Missing finetuned CLIP directory: {args.clip_ft}'

    print("Loading models...")
    processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to('cuda')

    model_orig = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model_orig.to('cuda')
    model_orig.eval()

    for param in model_orig.parameters():
        param.requires_grad = False

    clip_proc = CLIPProcessor.from_pretrained(args.clip_ft)
    clip = CLIPModel.from_pretrained(args.clip_ft)
    clip.to('cuda')
    clip.eval()

    print("Models loaded")

    print("Loading data...")

    name2spl = pd.read_csv(args.pseudolabels,
                   usecols=['name', 'spl']).set_index('name').spl.to_dict()
    name2bt = pd.read_csv(args.pseudolabels,
                   usecols=['name', 'building_type']).set_index('name').building_type.to_dict()

    ds = HashDS(args.hash_data, args.hash_metadata, name2spl, name2bt, neg_only=False)
    ds_test = HashDS(args.hash_data, args.hash_metadata, neg_only=True, spl='test')

    print("done")
    

if __name__ == "__main__":
    main()
