import multiprocessing
import os
from os.path import exists
from tqdm import tqdm
import yaml
import argparse
import pickle
import numpy as np
import random

import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim

from transformers import get_linear_schedule_with_warmup
from codet5_datasets.codesearchnet import CodeSearchNet
from models.codet5_summ import CodeT5Summ
from metrics.bleu import compute_bleu
from utils import set_seed


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def main(cfg):
    dataset_name = cfg.get('dataset', 'codexglue')
    assert dataset_name in ['csn', 'conala', 'codexglue']
    print(f"Dataset = {dataset_name}")
    if dataset_name == 'csn':
        root = cfg.get('root_csn', 'path/to/CSN/dataset')
    elif dataset_name == 'codexglue':
        root = cfg.get('root_codexglue', 'path/to/CodeXGlue/dataset')
    elif dataset_name == 'conala':
        root = cfg.get('root_conala', 'path/to/CoNaLa/dataset')
    gpu_id = cfg.get('gpu_id', 0)
    batch_size = cfg.get('batch_size', 32)
    epochs = cfg.get('epochs', 4)
    checkpoints_dir = cfg.get('checkpoints_dir', 'checkpoints')
    # num_workers = min(batch_size, multiprocessing.cpu_count() - 1)
    remove_func = cfg.get('remove_func', False)
    remove_comments = cfg.get('remove_comments', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    use_intent = cfg.get('use_intent', False)
    model_type = cfg.get('model_type', 'codebert')
    assert model_type in ['codebert', 'graphcodebert', 'codet5']
    print(f"Model used: {model_type}")
    dataset_path = cfg.get('dataset_pickle_path', 'datasets/saved_datasets')
    
    only_func = cfg.get('only_func', False)
    if model_type == 'codebert':
        if only_func:
            pretrained_model = "bert-base-uncased"
        else:
            pretrained_model = "microsoft/codebert-base"
    elif model_type == 'graphcodebert':
        print("Training GraphCodeBERT")
        pretrained_model = "microsoft/graphcodebert-base"
    elif model_type == 'codet5':
        pretrained_model = "Salesforce/codet5-base-multi-sum"
    
    print(f"Loading dataset: {remove_func} {only_func} {remove_comments} {adversarial_func}")
    if dataset_name in ['csn', 'codexglue']:
        if remove_keyword:
            fname = f'{dataset_path}/train_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_keyword.pkl'
        else:
            fname = f'{dataset_path}/train_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl'
        train_dataset = pickle.load(open(fname, 'rb'))
    elif dataset_name == 'conala':
        if use_intent:
            train_dataset = pickle.load(open(f'{dataset_path}/train_conala_intent_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl', 'rb'))
        else:
            train_dataset = pickle.load(open(f'{dataset_path}/train_conala_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl', 'rb'))

    model = CodeT5Summ(gpu_id=gpu_id)
    print(get_n_params(model.model))
    exit()
    # train_dataset = [model.tokenizer.cls_token+c+model.tokenizer.sep_token+d+model.tokenizer.sep_token for c, d in train_dataset]

    model.train()
    max_grad_norm = cfg.get('max_grad_norm', 1.0)
    # Define the optimizer
    params = model.parameters()
     # Prepare optimizer and schedule (linear warmup and decay)
    optimizer = optim.AdamW(params, lr=float(cfg.get('lr', 1e-4)), eps = cfg.get('adam_epsilon', 1e-8), weight_decay=cfg.get('weight_decay', 0.0))

    len_train = len(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, collate_fn=None, pin_memory=False)

    gradient_accumulation_steps = cfg.get('gradient_accumulation_steps', 1)
    warmup_steps = cfg.get('warmup_steps', 0)
    # t_total = len(train_dataset) // gradient_accumulation_steps * epochs
    t_total = len(train_loader) // gradient_accumulation_steps * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, t_total)

    if remove_keyword:
        checkpoint_path = f'{checkpoints_dir}/{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_keyword'
    else:
        checkpoint_path = f'{checkpoints_dir}/{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}'
    
    #List to store losses
    losses = []
    epoch = 1
    train_loss = []
    val_loss = []

    for epoch in range(1, epochs+1):
        running_loss = 0
        model.train()
        for batch, (code, desc) in enumerate(tqdm(train_loader)):
            code = list(code)
            desc = list(desc)
            loss = model.train_minibatch(code, desc)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            running_loss += loss.item()
            # Update parameters
            if (batch + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                # Zero the parameter gradients
                optimizer.zero_grad()
            # tqdm.write(torch.cuda.memory_summary(device=model.device))
            t_mem = torch.cuda.get_device_properties(0).total_memory
            r_mem = torch.cuda.memory_reserved(0)
            a_mem = torch.cuda.memory_allocated(0)
            f_mem = r_mem-a_mem  # free inside reserved
            msg = f"Total Memory = {t_mem}. Reserved Memory = {r_mem}. Allocated Memory = {a_mem}. Free Memory = {f_mem}"
            tqdm.write(msg)
            # if batch % 50 == 0:
            torch.cuda.empty_cache()
            # tqdm.write("Memory used = " + str(torch.cuda.memory_allocated(device=model.device)))
        final_loss = running_loss / len_train
        train_loss.append(final_loss)
        print(f"Training Losses = {train_loss}")
        if dataset_name in ['csn', 'codexglue']:
            # if remove_keyword:
            #     fname = f'{checkpoint_path}_keyword_{dataset_name}_{epoch}.pt'
            #     prev_fname = f'{checkpoint_path}_keyword_{dataset_name}_{epoch-1}.pt'
            # else:
            fname = f'{checkpoint_path}_{dataset_name}_{epoch}.pt'
            prev_fname = f'{checkpoint_path}_{dataset_name}_{epoch-1}.pt'
        elif dataset_name == 'conala':
            if use_intent:
                fname = f'{checkpoints_dir}/conala_intent_{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.pt'
                prev_fname = f'{checkpoints_dir}/conala_intent_{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch-1}.pt'
            else:
                fname = f'{checkpoints_dir}/conala_{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.pt'
                prev_fname = f'{checkpoints_dir}/conala_{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch-1}.pt'
        torch.save(model, fname)
        if exists(prev_fname):
            os.remove(prev_fname)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/config.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=42, type=int,
        help='seed for random number generation')
    args = parser.parse_args()
    # Set random seed for reproducibility
    set_seed(args)
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg)
