from argparse import ArgumentParser
from retrieval_utils.geom_retrieval import GeometricRetriver
from tqdm.auto import tqdm
import os
from PIL import Image
from glob import glob
import pandas as pd

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--building_type', '-b', type=str, required=True, help="name of building type, e.g. 'cathedral', 'mosque'")
    parser.add_argument('--images_folder', '-i', type=str, required=True, help="folder containing RGB images")
    parser.add_argument('--min_area', type=float, default=0.1, help="minimum area for geometric retrieval")
    parser.add_argument('--max_area', type=float, default=0.9, help="maximum area for geometric retrieval")
    return parser.parse_args()

def main():
    args = get_opts()

    assert os.path.exists(args.images_folder), f'Missing images directory: {args.images_folder}'

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

    df = pd.DataFrame(geo_scores).T.reset_index()
    df.columns = ['fn', 'score_area', 'score_min', 'score_med']
    df['base_fn'] = df.fn.apply(os.path.basename)
    area_cond = ((df.score_area > args.min_area) & (df.score_area < args.max_area))
    df['score'] = df.score_min + area_cond
    df = df.drop(columns=[c for c in df.columns if 'score_' in c])

    print(df.head())

    print("done")

if __name__ == "__main__":
    main()