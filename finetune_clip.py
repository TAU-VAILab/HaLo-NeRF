import pandas as pd
# import torch
# from tqdm.auto import tqdm
from argparse import ArgumentParser
import os
from PIL import Image
from sentence_transformers import InputExample
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers import losses
from torch.utils.data import DataLoader

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--pseudolabels', '-p', type=str, required=True, help="filename of pseudolabel csv file")
    parser.add_argument('--epochs', '-e', type=int, default=5, help="epochs")
    parser.add_argument('--lr', '-l', type=float, default=1e-6, help="learning rate")
    parser.add_argument('--batch_size', '-b', type=int, default=2 ** 7, help="batch size")
    parser.add_argument('--num_workers', '-n', type=int, default=20, help="number of dataloader workers")
    parser.add_argument('--output', '-o', type=str, default="data/clip_ckpt", help="output checkpoint directory")
    return parser.parse_args()

def row2ex(row):
    ps, fn = row.pseudolabel, row.fn
    img = Image.open(fn).convert('RGB')
    return InputExample(texts=[img, ps])

class DS(Dataset):
    
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        return row2ex(self.df.iloc[idx])

def main():

    args = get_opts()

    print("Loading pseudolabel table...")
    fn = args.pseudolabels
    assert os.path.exists(fn), f'Missing file: {fn}'
    df = pd.read_csv(fn)
    print("Pseudolabel table loaded")

    print(f"{len(df)} rows")
    df = df[df.pseudolabel.notna()].copy()
    print(f"Only using {len(df)} rows with non-empty pseudolabels")

    print("Train-test split:")
    print(df.spl.value_counts())

    df_train = df[df.spl == 'train'].copy()

    ds = DS(df_train)

    print("Loading CLIP model...")
    model = SentenceTransformer('clip-ViT-B-32')
    print("CLIP model loaded")

    dl = DataLoader(
        ds,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=model.smart_batching_collate,
        pin_memory=True)
    
    loss = losses.MultipleNegativesRankingLoss(model=model)

    print("Training model...")
    model.fit(
        train_objectives=[(dl, loss)],
        epochs=args.epochs,
        output_path=args.output,
        optimizer_params={'lr': args.lr}
    )

    out_fn = f'{args.output}/last'
    print("Saving final model to:", out_fn)
    model.save(out_fn)

    print("done")

if __name__ == "__main__":
    main()
