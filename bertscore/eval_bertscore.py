import argparse
import os
import yaml
from evaluate import load
import pandas as pd
import numpy as np


def main(cfg, args):
    bertscore = load("bertscore")
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    evaluation_model = "distilbert-base-uncased"
    if args.mtype == 'palm':
        fname = '../palm/results'
    elif args.mtype in ['llama2', 'codellama']:
        fname = '../llama/results/'
    if args.mtype in ['llama2', 'codellama']:
        if remove_func:
            fname = os.path.join(fname, f'obfusc_{args.mtype}_{args.parameters}.csv')
        elif adversarial_func:
            fname = os.path.join(fname, f'advers_{args.mtype}_{args.parameters}.csv')
        elif remove_keyword:
            fname = os.path.join(fname, f'remove_keyword_{args.mtype}_{args.parameters}.csv')
        elif only_func:
            fname = os.path.join(fname, f'no_body_{args.mtype}_{args.parameters}.csv')
        else:
            fname = os.path.join(fname, f'original_{args.mtype}_{args.parameters}.csv')
    
    print(f"Variant = {fname}")
    results = pd.read_csv(fname)
    reference = results['reference'].tolist()
    generated = results['generated_desc'].tolist()
    generated = [str(item).strip() for item in generated]

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
    parser.add_argument(
        '--mtype', default='incoder', type=str, choices=["incoder", "palm", "llama2", 'codellama'],
        help='Model to evaluate')
    parser.add_argument(
        '-p', '--parameters', default='7b', type=str, choices=['7b', '13b', '70b'],
        help='Number of Parameter in Llama2 Model')
    args = parser.parse_args()
    # Set random seed for reproducibility
    # set_seed(args)
    with open(f"../{args.config}") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)