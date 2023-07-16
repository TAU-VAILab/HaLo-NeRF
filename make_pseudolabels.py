import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from tqdm.auto import tqdm
from argparse import ArgumentParser
import os

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--input_table', '-i', type=str, required=True, help="filename of csv table containing image metadata")
    parser.add_argument('--output_table', '-o', type=str, required=True, help="csv filename to output to")
    return parser.parse_args()

def make_prompt(row):
    prompt = f"""
What architectural feature of {row['name']} is described in the following image? Write "unknown" if it is not specified.
Filename: {row.img_fn}
Caption: {row.caption}
Categories: {row.cats}
""".strip() + '\n'
    return prompt

def main():

    args = get_opts()

    print("Loading table...")
    fn = args.input_table
    assert os.path.exists(fn), f'Missing file: {fn}'
    df = pd.read_csv(fn)
    print("Table loaded")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Loading LM.... (device: {device})")
    model_id = "google/flan-t5-xl"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    model.to(device)
    model.eval()

    print("LM loaded")

    generate_kwargs = {
        'do_sample': False,
        'num_beams': 4,
        'length_penalty': 0,
        'early_stopping': True
    }
    BSZ = 2

    df['batch'] = df.index // BSZ

    tqdm.pandas(desc="Making prompts")
    df['prompt'] = df.progress_apply(make_prompt, axis=1)

    preds = []
    for _, subdf in tqdm(df.groupby('batch'), desc="Generating pseudo-labels"):
        prompts = subdf.prompt.to_list()
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.generate(**inputs.to('cuda'), **generate_kwargs)
        preds.append(tokenizer.batch_decode(outputs.cpu(), skip_special_tokens=True))
    
    preds_flat = pd.Series([y for x in preds for y in x])
    df['pseudolabel'] = preds_flat

    fn = args.output_table
    print("Saving to:", fn)
    if os.path.exists(fn):
        print(f"Warning: overwriting existing file {fn}")
    df.to_csv(fn, index=False)


    print("done")

if __name__ == "__main__":
    main()