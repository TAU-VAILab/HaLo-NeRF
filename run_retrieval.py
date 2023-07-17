from argparse import ArgumentParser
from retrieval_utils.geom_retrieval import GeometricRetriver
from tqdm.auto import tqdm, trange
import os
from PIL import Image
from glob import glob
import pandas as pd
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import torch
import numpy as np

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--building_type', '-b', type=str, required=True, help="name of building type, e.g. 'cathedral', 'mosque'")
    parser.add_argument('--images_folder', '-i', type=str, required=True, help="folder containing RGB images")
    parser.add_argument('--rgb_reconstruction_folder', '-r', type=str, required=True, help="folder containing RGB reconstruction images")
    parser.add_argument('--min_area', type=float, default=0.1, help="minimum area for geometric retrieval")
    parser.add_argument('--max_area', type=float, default=0.9, help="maximum area for geometric retrieval")
    parser.add_argument('--clipseg_threshold', type=float, default=0.5, help="threshold for clipseg in occlusion scoring")
    return parser.parse_args()

def main():
    args = get_opts()

    assert os.path.exists(args.images_folder), f'Missing images directory: {args.images_folder}'
    assert os.path.exists(args.rgb_reconstruction_folder), f'Missing reconstructed images directory: {args.rgb_reconstruction_folder}'

    bt = args.building_type
    print("Building type:", bt)

    fns = glob(os.path.join(args.images_folder, '*'))
    print(len(fns), 'image files found')

    print("Running geometric retrieval...")
    geo_scores = {}
    gr = GeometricRetriver(prompt=bt)
    for fn in tqdm(fns):
        img = Image.open(fn).convert('RGB')
        scores = gr.process(img)
        geo_scores[fn] = scores

    df_geo = pd.DataFrame(geo_scores).T.reset_index()
    df_geo.columns = ['fn', 'score_area', 'score_min', 'score_med']
    df_geo['base_fn'] = df_geo.fn.apply(os.path.basename)
    area_cond = ((df_geo.score_area > args.min_area) & (df_geo.score_area < args.max_area))
    df_geo['score'] = df_geo.score_min + area_cond
    df_geo = df_geo.drop(columns=[c for c in df_geo.columns if 'score_' in c])
    df_geo['ID'] = df_geo.fn.str.split('/').apply(lambda x: x[-1].split('.')[0])
    df_geo.ID = df_geo.ID.apply(lambda x: '0' * (4 - len(x)) + x)

    print("Running occlusion scoring...")
    print("Loading CLIP...")
    proc = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")
    model.to('cuda')
    model.eval()

    def occ_score_pair(img, img_nerf):
        inp = proc(text=[bt] * 2, images=[img, img_nerf], padding="max_length", return_tensors="pt").to('cuda')
        with torch.no_grad():
            out = model(**inp)
            S = out.logits.sigmoid().cpu().numpy()
        s0, s1 = S[0] > args.clipseg_threshold, S[1] > args.clipseg_threshold
        score = 0. if s1.sum() == 0 else (1 - ((~s0) & s1).sum() / s1.sum())
        M = (s0 & s1) + 2 * ((~s0) & s1)
        return score, M
    
    nerf_fns = glob(os.path.join(args.rgb_reconstruction_folder, '*'))
    print("Using", len(nerf_fns), "RGB reconstruction images")

    orig_df = pd.DataFrame({'fn': fns})
    nerf_df = pd.DataFrame({'fn': nerf_fns})
    for df_ in (orig_df, nerf_df):
        df_['ID'] = df_.fn.str.split('/').apply(lambda x: x[-1].split('.')[0])
        df_.ID = df_.ID.apply(lambda x: '0' * (4 - len(x)) + x)
    df_occ = pd.merge(orig_df, nerf_df, on=['ID'],
              how='outer', indicator=True, suffixes=('_orig', '_nerf'))
    df_occ['score'] = np.nan

    for i in trange(len(df_occ)):
        row = df_occ.iloc[i]
        if row._merge == 'both':
            img = Image.open(row.fn_orig).convert('RGB')
            img_nerf = Image.open(row.fn_nerf).convert('RGB')
            s, M = occ_score_pair(img, img_nerf)
            df_occ.score.iloc[i] = s

    print("Combining scores...")
    id2geo = df_geo.set_index('ID').score.to_dict()
    df = df_occ.copy().rename(columns={'score': 'occ_score'})
    df['geo_score'] = df.ID.map(id2geo)
    df['score'] = df.geo_score + df.occ_score

    print("df_geo")
    print(df_geo.head())
    print("df_occ")
    print(df_occ.head())

    print("done")

if __name__ == "__main__":
    main()