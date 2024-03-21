import os
from tqdm import tqdm
import json
import argparse
import yaml
import random
import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import nltk

from transformers import RobertaTokenizer


def get_tokens(tokenizer, tokens):
    tokens = tokenizer.tokenize(tokens.lower())
    new_tokens = []
    gc = ['Ġ', 'Ċ']
    punctuations = [',', '.', '(', ')', '{', '}', '[', ']', ':']
    remove_tokens = gc + punctuations
    for t in tokens:
        for rt in remove_tokens:
            t = t.replace(rt, '')
        if t:
            new_tokens.append(t)
    return tokens


def compute_bleu(cfg):
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    pretrained_model="Salesforce/codet5-base-multi-sum"
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    fname = 'results/palm.csv'
    if remove_func:
        fname = 'results/palm_obfusc.csv'
    if adversarial_func:
        fname = 'results/palm_advers.csv'
    if remove_keyword:
        fname = 'results/palm_remove_keyword.csv'
    if only_func:
        fname = 'results/palm_no_body.csv'
    print(f"Variant = {fname}")
    results = pd.read_csv(fname)


    reference = results['reference'].tolist()
    generated = results['generated_desc'].tolist()
    generated = [str(item).strip() for item in generated]
    refs = []
    gens = []

    for data in tqdm(reference):
        toks = get_tokens(tokenizer, data)
        # toks = data.split()
        refs.append([toks])
    for data in tqdm(generated):
        toks = get_tokens(tokenizer, data)
        # toks = data.split()
        gens.append(toks)

    max_order = 4
    smooth = True
    print(f"Computing BLEU Score with max order={max_order} and smoothing set to {smooth}") 
    score = 0
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    for ref, gen in zip(refs, gens):
        if smooth:
            score += nltk.translate.bleu_score.sentence_bleu(ref, gen, smoothing_function=chencherry.method2)
        else:
            score += nltk.translate.bleu_score.sentence_bleu(ref, gen, smoothing_function=chencherry.method0)
    score = 100.0 * score / (len(gens))
    print(f"BLEU Score = {score}")



def main(cfg, args):
    compute_bleu(cfg)


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
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)