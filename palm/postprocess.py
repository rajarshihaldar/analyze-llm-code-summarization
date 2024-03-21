import os
import argparse
import json
import yaml
from tqdm import tqdm
import pandas as pd


# Takes the output dump from inference.py and seperates the fields into a Pandas dataframe
def seperate_fields(results):
    fields = ['idx', 'prompt', 'reference', 'generated_desc']
    df = pd.DataFrame(columns=fields)

    for i, example in enumerate(results, 1):
        split1 = example.split('<Reference Desc>')
        prompt = split1[0].replace('<Prompt>', '').strip()
        split2 = split1[1].split('<Generated Desc>')
        reference = split2[0].replace('<Reference Desc>', '').strip()
        generated_desc = split2[1].replace('<Generated Desc>', '').strip()
        row = {'idx': i, 'prompt': prompt, 'reference': reference, 'generated_desc': generated_desc}
        df = df.append(row, ignore_index = True)
    return df


def main(cfg, args):
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    fname = 'results/palm.txt'
    if remove_func:
        fname = 'results/palm_obfusc.txt'
    if adversarial_func:
        fname = 'results/palm_advers.txt'
    if remove_keyword:
        fname = 'results/palm_remove_keyword.txt'
    if only_func:
        fname = 'results/palm_no_body.txt'
    with open(fname, 'r') as f:
        results = f.read()

    results = results.split('<Example Start>')[1:]
    print(f"{len(results)} examples in results.")

    df = seperate_fields(results)
    df.to_csv(f'{os.path.splitext(fname)[0]}.csv')
    print(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/1.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=42, type=int,
        help='seed for random number generation')
    args = parser.parse_args()
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)