from tqdm.auto import tqdm, trange
from torch import nn
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import CLIPProcessor, CLIPModel
from collections import namedtuple
from torchvision.transforms import Resize
from argparse import ArgumentParser
import os
import pandas as pd
from seg_utils.hash_data import HashDS
from seg_utils.crop_data import search_crops, CropDS
from seg_utils.plurals import to_plur_aug
import torch
from torch.utils.data import DataLoader
import json

def get_opts():
    parser = ArgumentParser()
    
    parser.add_argument('--clip_ft', '-c', type=str, default="data/clip_ckpt/0_CLIPModel", help="directory of fine-tuned CLIP checkpoint")
    parser.add_argument('--data_dir', '-d', type=str, default="data/wikiscenes", help="directory WikiScenes data is stored in")
    parser.add_argument('--crop_metadata', '-cm', type=str, default="data/seg_crop_metadata.csv", help="crop metadata filename")
    parser.add_argument('--hash_metadata', '-hm', type=str, default="data/seg_hash_metadata.csv", help="hash metadata filename")
    parser.add_argument('--hash_data', '-hd', type=str, default="data/hashdata", help="hash data (correspondence-based data) directory")
    parser.add_argument('--pseudolabels', '-p', type=str, default="data/pseudolabels.csv", help="filename of pseudolabel csv file")
    parser.add_argument('--crop_batch_size', '-cb', type=int, default=4, help="crop data batch size")
    parser.add_argument('--epochs', '-e', type=int, default=10, help="epochs")
    parser.add_argument('--lr', '-l', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--grad_acc_steps', '-g', type=int, default=1, help="gradient accumulation steps")
    parser.add_argument('--n_crops', '-nc', type=int, default=2, help="number of crops to try per step")
    parser.add_argument('--L0', '-L0', type=float, default=1.0, help="loss coefficient 0")
    parser.add_argument('--L1', '-L1', type=float, default=1.0, help="loss coefficient 1")
    parser.add_argument('--L2', '-L2', type=float, default=1.0, help="loss coefficient 2")
    parser.add_argument('--L3', '-L3', type=float, default=1.0, help="loss coefficient 3")
    parser.add_argument('--output', '-o', type=str, default="data/seg_ckpt", help="output checkpoint directory")
    parser.add_argument('--output_losses', '-ol', type=str, default="data/seg_losses.json", help="output losses log")

    return parser.parse_args()

def main():
    args = get_opts()

    assert os.path.exists(args.crop_metadata), f'Missing crop metadata file: {args.crop_metadata}'
    assert os.path.exists(args.hash_metadata), f'Missing hash metadata file: {args.hash_metadata}'
    assert os.path.exists(args.hash_data), f'Missing hash data directory: {args.hash_data}'
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

    hash_ds = HashDS(args.hash_data, args.hash_metadata, name2spl, name2bt, neg_only=False)
    # hash_ds_test = HashDS(args.hash_data, args.hash_metadata, name2spl, name2bt, neg_only=True, spl='test')

    crop_ds = CropDS(args.crop_metadata)

    print("Setting up data loaders...")

    CropMinibatch = namedtuple("Minibatch", "imgs labels coords coords_ gts")

    def crop_collate(B):
        inp = processor(
                images=[r.img.crop(r.coords) for r in B],
                text = [r.label for r in B],
                padding=True,
                return_tensors="pt").to('cuda')
        with torch.no_grad():
            gt = model_orig(**inp).decoder_output.logits.sigmoid()
        return CropMinibatch(
            [r.img for r in B],
            [to_plur_aug(r.label) for r in B],
            [r.coords for r in B],
            [r.coords_ for r in B],
            gt
        )
    
    crop_dl = DataLoader(
        crop_ds,
        batch_size=args.crop_batch_size,
        num_workers=0,
        collate_fn=crop_collate
    )

    hash_dl = DataLoader(
        hash_ds,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda x: x[0]
    )

    print("Starting training...")

    h, w = 352, 352
    rsz = Resize((h, w), antialias=True)
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = args.epochs
    lr = args.lr
    grad_acc_steps = args.grad_acc_steps
    n_crops = args.n_crops
    L0, L1, L2, L3 = args.L0, args.L1, args.L2, args.L3

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    losses0 = []
    losses1 = []
    losses2 = []
    losses3 = []

    for i in trange(epochs):
        for j, x in enumerate(tqdm(hash_dl)):
            
            if j % grad_acc_steps == 0:

                optimizer.step()
                optimizer.zero_grad()
                
            label_orig = x.label
            label = to_plur_aug(x.label)
                
            inp = processor(
                images=x.img_,
                text = label,
                padding=True,
                return_tensors="pt").to('cuda')
            out = model(**inp).decoder_output.logits
            gt = torch.tensor(x.gt_).to('cuda')
            mask = torch.tensor(x.mask_).to('cuda')
            mask_flat = mask.ravel() > 0
            loss = loss_fn(out.ravel()[mask_flat], gt.ravel()[mask_flat])
            loss = loss * L0
            losses0.append(loss.item())

            ent = loss_fn(out, out.sigmoid())
            loss_e = ent.mean()
            loss += loss_e * L1
            losses1.append(loss_e.item())
            
            
            c = search_crops(x.img_, label_orig, n_crops=n_crops)
            out_crop = out[c.y0:c.y1, c.x0:c.x1]
            out_crop = rsz(out_crop[None])[0]
            inp_crop = processor(
                images=c.C,
                text = label,
                padding=True,
                return_tensors="pt").to('cuda')
            # out_crop = model(**inp_crop).decoder_output.logits
            with torch.no_grad():
                gt_crop = model_orig(**inp_crop).decoder_output.logits.sigmoid()
            loss_c = loss_fn(out_crop.ravel(), gt_crop.ravel())
            loss += loss_c * L2
            losses2.append(loss_c.item())
            
            
            R = next(iter(crop_dl))
            new_gt = R.gts
            inp_new = processor(
                images=R.imgs,
                text = R.labels,
                padding=True,
                return_tensors="pt").to('cuda')
            out_new = model(**inp_new).decoder_output.logits
            out_new_crops = []
            for ON, co in zip(out_new, R.coords_):
                x0, y0, x1, y1 = co
                ON_c = ON[y0:y1, x0:x1]
                ON_c = rsz(ON_c[None])[0]
                out_new_crops.append(ON_c)
            out_new_crops = torch.stack(out_new_crops) # (b, 352, 352)
            
            loss_new = loss_fn(out_new_crops.ravel(), new_gt.ravel())
            loss += loss_new * L3
            losses3.append(loss_new.item())
            
            
            loss.backward()

            losses.append(loss.item())
            
        optimizer.step()
        optimizer.zero_grad()


    print("Training finished")

    print("Saving losses to:", args.output_losses)
    if os.path.exists(args.output_losses):
        print(f"Warning: {args.output_losses} exists; overwriting")
    losses_obj = { 'losses': losses, 'losses0': losses0, 'losses1': losses1, 'losses2': losses2, 'losses3': losses3 }
    with open(args.output_losses, 'w') as f:
        json.dump(losses_obj, f, indent=4)

    print("Saving trained model to:", args.output)
    if os.path.exists(args.output):
        print(f"Warning: {args.output} exists; overwriting")
    
    model.save_pretrained(args.output)

    print("done")
    

if __name__ == "__main__":
    main()
