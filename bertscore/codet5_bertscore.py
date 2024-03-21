import argparse
import os
import yaml
from evaluate import load
import pandas as pd
import numpy as np
import json


def main(cfg, args):
    bertscore = load("bertscore")
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    evaluation_model = "distilbert-base-uncased"
    path = '../codet5/results'
    
    if remove_keyword:
        fname = f'codet5_codexglue_{remove_func}_{only_func}_{adversarial_func}_{adversarial_func}_keyword_8.json'
    else:
        fname = f'codet5_codexglue_{remove_func}_{only_func}_{adversarial_func}_{adversarial_func}_8.json'
    results = json.load(open(os.path.join(path, fname)))
    
    print(f"Variant = {fname}")
    reference = []
    generated = []

    for item in results['results']:
        reference.append(item['ref'])
        generated.append(item['gen'])

    results = bertscore.compute(predictions=generated, references=reference, model_type=evaluation_model)
    precision = np.mean(results['precision'])
    recall = np.mean(results['recall'])
    f1 = np.mean(results['f1'])

    print(f"Precision = {precision}")
    print(f"Recall = {recall}")
    print(f"F1 = {f1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/1.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=42, type=int,
        help='seed for random number generation')
    args = parser.parse_args()
    # Set random seed for reproducibility
    # set_seed(args)
    with open(f"../{args.config}") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)