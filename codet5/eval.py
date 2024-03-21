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
from scipy import stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import torch
import torch.utils.data as torch_data
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import nltk

from transformers import RobertaTokenizer, AutoTokenizer

from codet5_datasets.codesearchnet import CodeSearchNet
from models.codet5_summ import CodeT5Summ

from utils import compute_mrr, compute_recall_at_k, set_seed
from metrics.bleu import compute_bleu


plt.style.use('seaborn')
sns.color_palette("colorblind")
WIDTH = 455.24408
KEYWORD = 256
# First line has variants with comments and second line has variants without
# variants = [(False, False, False, False), (False, False, True, False), (True, False, False, False), (False, False, False, True), (KEYWORD, False, False, False), (False, True, False, False)]
variants = [(False, False, False, False), (False, False, True, False), (True, False, True, False), (False, False, True, True), (KEYWORD, False, True, False), (False, True, True, False)]

variant_mapping = {f"{str((False, False, False, False))}": "Original Function Names",
f"{str((False, False, True, False))}":"Original Function Names (without Comments)",
f"{str((KEYWORD, False, False, False))}":"No Code Structure",
f"{str((KEYWORD, False, True, False))}":"No Code Structure (without Comments)",
f"{str((True, False, False, False))}":"Obfuscated Function Names",
f"{str((True, False, True, False))}":"Obfuscated Function Names (without Comments)",
f"{str((False, True, False, False))}":"No Function Body",
f"{str((False, True, True, False))}":"No Function Body (without Comments)",
f"{str((False, False, False, True))}":"Adversarial Function Names",
f"{str((False, False, True, True))}":"Adversarial Function Names (without Comments)"}
def set_size(width, fraction=1):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float
            Document textwidth or columnwidth in pts
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy

    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = width * fraction

    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio

    fig_dim = (fig_width_in, fig_height_in)

    return fig_dim


def get_result_path(cfg):
    result_path = cfg.get('result_path', 'results')
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    remove_comments = cfg.get('remove_comments', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    use_intent = cfg.get('use_intent', False)
    model_type = cfg.get('model_type', 'codebert')
    dataset_name = cfg.get('dataset', 'codexglue')
    use_pretrained = cfg.get('use_pretrained', False)
    assert dataset_name in ['csn', 'conala', 'codexglue']
    epoch = cfg.get('eval_epoch', 1)
    if not use_pretrained:
        model_type = f"{model_type}_finetuned"
    if dataset_name == 'conala':
        result_path = os.path.join(result_path, 'conala')
        if use_intent:
            result_path = os.path.join(result_path, f'{model_type}_{dataset_name}_intent_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.json')
        else:
            result_path = os.path.join(result_path, f'{model_type}_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.json')
    else:
        if remove_keyword:
            result_path = os.path.join(result_path, f'{model_type}_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_keyword_{epoch}.json')
        else:
            result_path = os.path.join(result_path, f'{model_type}_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.json')
    return result_path


def translate(args, cfg):
    epoch = cfg.get('eval_epoch', 1)

    dataset_name = cfg.get('dataset', 'csn')
    gpu_id = cfg.get('gpu_id', 0)
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    remove_comments = cfg.get('remove_comments', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    use_intent = cfg.get('use_intent', False)
    model_type = cfg.get('model_type', 'codebert')
    assert model_type in ['codebert', 'graphcodebert', 'codet5']
    assert dataset_name in ['csn', 'conala', 'codexglue']
    batch_size = cfg.get('eval_batch_size', 8)
    use_pretrained = cfg.get('use_pretrained', False)

    dataset_path = cfg.get('dataset_pickle_path', 'datasets/saved_datasets')

    use_random_samples = False


    if dataset_name in ['csn', 'codexglue']:
        if remove_keyword:
            fname = f'{dataset_path}/test_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_keyword.pkl'
        else:
            fname = f'{dataset_path}/test_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl'
        test_dataset = pickle.load(open(fname, 'rb'))
    elif dataset_name == 'conala':
        if use_intent:
            test_dataset = pickle.load(open(f'{dataset_path}/test_conala_intent_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl', 'rb'))
        else:
            test_dataset = pickle.load(open(f'{dataset_path}/test_conala_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl', 'rb'))
    
    
    checkpoints_dir = cfg.get('checkpoints_dir', 'checkpoints')

    if remove_keyword:
        checkpoint_path = f'{checkpoints_dir}/{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_keyword'
    else:
        checkpoint_path = f'{checkpoints_dir}/{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}'

    if dataset_name in ['csn', 'codexglue']:
        # if remove_keyword:
        #     model_fname = f'{checkpoint_path}_keyword_{dataset_name}_{epoch}.pt'
        # else:
        model_fname = f'{checkpoint_path}_{dataset_name}_{epoch}.pt'
    elif dataset_name == 'conala':
        if use_intent:
            model_fname = f'{checkpoints_dir}/conala_intent_{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.pt'
        else:
            model_fname = f'{checkpoints_dir}/conala_{model_type}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.pt'

    if use_pretrained:
        model = CodeT5Summ(gpu_id=gpu_id)
    else:
        model = torch.load(model_fname)

    model.eval()
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device = {device}")
    model.set_device(gpu_id)


    if use_random_samples:
        indices = pickle.load(open('../../llm_analysis/palm/random_indices.pkl', 'rb'))
        random_indices = defaultdict(bool)
        for ind in indices:
            random_indices[ind] = True
        new_dataset = []
        for i, item in enumerate(test_dataset):
            if not random_indices[i]:
                continue
            code, desc = item
            new_dataset.append((code, desc))
        test_dataset = new_dataset

    num_examples = len(test_dataset)
    print(f"Dataset = {dataset_name}")
    
    if batch_size > 1:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, collate_fn=None, pin_memory=False)
    else:
        test_loader = test_dataset

    results = {}
    results["results"] = []
    codes = []
    refs = []
    gens = []
    print(f"Config = {os.path.basename(args.config)}")
    print(f"Model={remove_func}_{only_func}_{remove_comments}_{adversarial_func}_{epoch}.")
    for i, item in enumerate(tqdm(test_loader), 1):
        code, ref = item
        if batch_size == 1:
            gen = model.summarize(code)
            codes.append(code)
            refs.append(ref)
            gens.append(gen)
        else:
            gen = model.batch_summarize(code)
            codes.extend(code)
            refs.extend(ref)
            gens.extend(gen)

    
    for code, ref, gen in zip(codes, refs, gens):
        output = {}
        output["code"] = code
        output["ref"] = ref
        output["gen"] = gen
        results["results"].append(output)

    result_path = get_result_path(cfg)
    json.dump(results, open(result_path, 'w'), indent=4)


# # Get RoBERTa tokens
# def get_tokens(tokenizer, tokens):
#     tokens = tokenizer.tokenize(tokens.lower())
#     new_tokens = []
#     gc = ['Ġ', 'Ċ']
#     punctuations = [',', '.', '(', ')', '{', '}', '[', ']', ':']
#     remove_tokens = gc + punctuations
#     for t in tokens:
#         for rt in remove_tokens:
#             t = t.replace(rt, '')
#         if t:
#             new_tokens.append(t)
#     return tokens

# Get Llama 2 tokens
def get_tokens(tokenizer, tokens):
    tokens = tokenizer.tokenize(tokens.lower())
    return tokens


def save_bleu(cfg):
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    refs = []
    gens = []
    pretrained_model="Salesforce/codet5-base-multi-sum"
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    for data in tqdm(results['results']):
        r_tokens = data['ref'].split()
        g_tokens = data['gen'].split()
        refs.append([r_tokens])
        gens.append(g_tokens)

    max_order = 4
    smooth = True
    use_custom_bleu = False
    print(f"Computing BLEU Score with max order={max_order} and smoothing set to {smooth}")
    if use_custom_bleu:
        score = compute_bleu(refs, gens, max_order=max_order, smooth=smooth)[0]
    else:
        score = 0
        chencherry = nltk.translate.bleu_score.SmoothingFunction()
        for ref, gen in zip(refs, gens):
            if smooth:
                score += nltk.translate.bleu_score.sentence_bleu(ref, gen, smoothing_function=chencherry.method2)
            else:
                score += nltk.translate.bleu_score.sentence_bleu(ref, gen, smoothing_function=chencherry.method0)
        score = 100.0 * score / (len(gens))
    print(f"BLEU Score = {score}")

    final_output = {}
    final_output["results"] = results["results"]
    final_output["bleu_score"] = score
    # json.dump(final_output, open(result_path, 'w'), indent=4)

    # df = pd.DataFrame(results)
    # df.to_excel(f"{os.path.splitext(result_path)[0]}.xlsx")


def view_results(cfg):
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    for data in tqdm(results['results'][:100]):
        code = data['code']
        if '#' in code:
            print(code)
            print()


def scatter_plots(x, y):
    fig, ax = plt.subplots(figsize=set_size(WIDTH))
    plt.scatter(x, y)
    ax.legend(loc='best')
    plt.savefig(f"plots/scatter/refcode_vs_refgen.pdf", format='pdf', bbox_inches='tight')


def plot_histogram(x, title, fname, xlabel, bins=100, only_stats=False):
    if not only_stats:
        weights = np.ones_like(x) / len(x)
        fig, ax = plt.subplots()
        # ax.hist(x=x, bins=bins, density=True, histtype='bar')
        ax.hist(x=x, bins=bins, density=False, histtype='bar', weights=weights)
        trans = ax.get_xaxis_transform()
        ax.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
        plt.text(x.mean()+1, .2, f"Mean = {'{:.2f}'.format(x.mean())}", transform=trans, color='w')
        # ax.legend(prop={'size': 10}, loc='upper right')
        ax.set_ylabel('Fraction of Examples', fontsize=14)
        ax.set_xlabel(xlabel, fontsize=14)
        title = title.split(':')
        suptitle = title[0]
        title = title[1]
        plt.suptitle(suptitle, fontsize=20)
        plt.title(title, fontsize=16)
        # ax.set_title(title)
        plt.savefig(f"plots/hists/{fname}.pdf")
        plt.close()
    mean = np.mean(x)
    median = np.median(x)
    if bins == 100:
        with open(f"plots/hists/{fname}_stats.txt", "w") as file:
            file.write(title+"\n")
            file.write(f"Mean = {mean}\n")
            file.write(f"Median = {median}\n")
            file.write(f"Mode = {st.mode(x)}\n")
            file.write(f"Standard Deviation = {np.std(x)}\n")
            file.write(f"Variance = {np.var(x)}\n")
    return mean, median


def get_token_overlap(desc, code):
    overlap = 0.0
    for tok in desc:
        if tok in code:
            overlap += 1
    overlap /= len(desc)
    return overlap


def save_all_bleu(cfg):
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    results["refcode"] = []
    results["refgen"] = []
    results["gencode"] = []
    refs = []
    gens = []
    pretrained_model="Salesforce/codet5-base-multi-sum"
    # tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    cache_dir = '/scratch/rhaldar2/huggingface/'
    model_size = '7b'
    tokenizer = AutoTokenizer.from_pretrained(f'meta-llama/Llama-2-{model_size}-chat-hf',
        cache_dir=cache_dir)
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    smooth = True
    weights = (1,)
    weights = (1,)
    refcode = []
    gencode = []
    refgen = []
    # token_overlap values
    refcode_to = []
    gencode_to = []
    refgenbleu1 = []
    for data in tqdm(results['results']):
        code = get_tokens(tokenizer, data['code'])
        ref = get_tokens(tokenizer, data['ref'])
        gen = get_tokens(tokenizer, data['gen'])
        refcode_to.append(get_token_overlap(ref, code))
        gencode_to.append(get_token_overlap(gen, code))
        if smooth:
            refgen.append(nltk.translate.bleu_score.sentence_bleu([ref], gen, 
                smoothing_function=chencherry.method2))
            refgenbleu1.append(nltk.translate.bleu_score.sentence_bleu([ref], gen, 
                smoothing_function=chencherry.method5, weights=weights))
            refcode.append(nltk.translate.bleu_score.sentence_bleu([ref], code, 
                smoothing_function=chencherry.method5, weights=weights))
            gencode.append(nltk.translate.bleu_score.sentence_bleu([gen], code, 
                smoothing_function=chencherry.method5, weights=weights))
        else:
            refgen.append(nltk.translate.bleu_score.sentence_bleu([ref], gen, 
                smoothing_function=chencherry.method0, weights=weights))
            refcode.append(nltk.translate.bleu_score.sentence_bleu([ref], code, 
                smoothing_function=chencherry.method0, weights=weights))
            gencode.append(nltk.translate.bleu_score.sentence_bleu([gen], code, 
                smoothing_function=chencherry.method0, weights=weights))

    results["refcode"] = refcode
    results["refgen"] = refgen
    results["gencode"] = gencode
    results["refgenbleu1"] = refgenbleu1
    results["refcode_to"] = refcode_to
    results["gencode_to"] = gencode_to

    json.dump(results, open(result_path, 'w'), indent=4)


def process_all_bleus(results, scoring_func="bleu1"):
    assert scoring_func in ["bleu1", "token_fraction"]
    refgen = results["refgen"]
    if scoring_func == "bleu1":
        refcode = results["refcode"]
        gencode = results["gencode"]
    elif scoring_func == "token_fraction":
        refcode = results["refcode_to"]
        gencode = results["gencode_to"]
    refgenbleu1 = results["refgenbleu1"]
    refcode = 100.0 * np.array(refcode)
    refgen = 100.0 * np.array(refgen)
    gencode = 100.0 * np.array(gencode)
    return refcode, refgen, gencode, refgenbleu1


def get_variant_name(cfg):
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    remove_comments = cfg.get('remove_comments', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    if remove_keyword:
        remove_func = KEYWORD
    variant = (remove_func, only_func, remove_comments, adversarial_func)
    variant_name = variant_mapping[str(variant)].replace(' ', '_')
    return variant_name


def create_stacked_hist(x , range=(0, 100), bins=[0, 20, 40, 60, 80]):
    hx = {}
    for b in bins:
        hx[b] = 0
    for i in x:
        for b in bins[::-1]:
            if i >= b:
                hx[b] += 1
                break
    n_examples = len(x)
    for k, v in hx.items():
        hx[k] = 100.0 * v / n_examples
    return hx



def plot_stacked_hist(refcode, gencode, title, fname, xlabel):
    refh = create_stacked_hist(refcode)
    genh = create_stacked_hist(gencode)
    colors = sns.color_palette("crest", n_colors=5)
    dims = set_size(WIDTH)
    dims = (dims[0], 2)
    fig, ax = plt.subplots(figsize=dims)
    bars = []
    for r, g in zip(refh.values(), genh.values()):
        bars.append([r, g])
    bars = np.array(bars)
    bottom = np.zeros(bars[0].shape)
    # x_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
    x_labels = ["80-100", "60-80", "40-60", "20-40", "0-20"]
    y_labels = ["Reference", "Generated"]
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())

    bars = np.flip(bars, 0)

    Y_axis = [0, 1]
    for i, bar in enumerate(bars):
        lab = x_labels[i]
        if i == 0:
            ax.barh(y=Y_axis, height=0.4, width=bar, color=colors[len(x_labels)-1-i], label=lab)
        else:
            ax.barh(y=Y_axis, height=0.4, width=bar, left=bottom, color=colors[len(x_labels)-1-i], label=lab)
        bottom += bar

    plt.yticks(Y_axis, y_labels) 
    ax.set_xlabel("Fraction of Examples")
    ax.set_ylabel("Type of Description:\nReference/Generated")
    # plt.suptitle(r"Distribution of $p_{copy}$")

    fig.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height*0.9])

    # Put a legend to the right of the current axis
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
    legend_str = r'$p_{copy}$'
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5),title=legend_str)

    plt.savefig(f"plots/hists/{fname}.pdf")


def simple_histograms(cfg, scoring_func="bleu1", only_stats=False):
    assert scoring_func in ["bleu1", "token_fraction"]
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    refcode, refgen, gencode, _ = process_all_bleus(results, scoring_func=scoring_func)

    variant_fname = get_variant_name(cfg)
    os.makedirs(f"plots/hists/{variant_fname}", exist_ok=True)

    if scoring_func == "bleu1":
        scoring_str = "BLEU-1"
        bins = [0, 1, 36, 40, 45, 101]
    elif scoring_func == "token_fraction":
        scoring_str = r'$n_{copy}$'
        bins = [0, 1, 20, 40, 60, 80, 101]
    
    # scatter_plots(refcode, refgen)
    if not only_stats:
        plot_stacked_hist(refcode, gencode, title=f'Distribution of ' + scoring_str + ':Between Reference Descripion and Code',
            fname=f"{variant_fname}/stacked_hist", xlabel=scoring_str)
        exit()

    refcode_mean, refcode_median = plot_histogram(refcode, title=f'Distribution of ' + scoring_str + ':Between Reference Descripion and Code',
        fname=f"{variant_fname}/refcode_{scoring_func}", xlabel=scoring_str, only_stats=only_stats)
    plot_histogram(refcode, title=f'Distribution of ' + scoring_str + ':Between Reference Descripion and Code',
        fname=f"{variant_fname}/refcode_{scoring_func}_bins", xlabel=scoring_str,
        bins=bins, only_stats=only_stats)
    gencode_mean, gencode_median = plot_histogram(gencode, title=f'Distribution of ' + scoring_str + ':Between Generated Descripion and Code',
        fname=f"{variant_fname}/gencode_{scoring_func}", xlabel=scoring_str, only_stats=only_stats)
    plot_histogram(gencode, title=f'Distribution of ' + scoring_str + ':Between Generated Descripion and Code',
        fname=f"{variant_fname}/gencode_{scoring_func}_bins", xlabel=scoring_str,
        bins=bins, only_stats=only_stats)
    plot_histogram(refgen, title='Distribution of BLEU-4:Between Reference and Generated Description',
        fname=f"{variant_fname}/refgen", xlabel="BLEU-4")
    plot_histogram(refgen, title='Distribution of BLEU-4:Between Reference and Generated Description',
        fname=f"{variant_fname}/refgen_bins", xlabel="BLEU-4", bins=[0, 1, 10, 20, 30, 40, 101], only_stats=only_stats)
    return refcode_mean, refcode_median, gencode_mean, gencode_median, refcode, gencode


# x-axis is refgen and y-axis is refcode/gencode
def get_bucketed_data(results, scoring_func="bleu1"):
    assert scoring_func in ["bleu1", "token_fraction"]
    refcode, refgen, gencode, _ = process_all_bleus(results, scoring_func=scoring_func)
    examples = results["results"]
    bucket_data = {}
    bucket_labels = ['0', '0-10', '10-20', '20-30', '30-40', '>40']
    bucket_lower_limits = [-np.inf, 0, 10, 20, 30, 40]
    for l, t in zip(bucket_labels, bucket_lower_limits):
        bucket_data[t] = {}
        bucket_data[t]["lower_limit"] = t
        bucket_data[t]["label"] = l
        bucket_data[t]["examples"] = []
    for example, rc, rg, gc in zip(examples, refcode, refgen, gencode):
        data = {}
        data["code"] = example["code"]
        data["ref"] = example["ref"]
        data["gen"] = example["gen"]
        data["refcode"] = rc
        data["refgen"] = rg
        data["gencode"] = gc
        for limit in bucket_lower_limits[::-1]:
            if rg > limit:
                bucket_data[limit]["examples"].append(data)
                break
    for limit in bucket_data:
        bucket_data[limit]["size"] = len(bucket_data[limit]["examples"])
    return bucket_data


# x-axis is refcode/gencode and y-axis is refgen
def get_reverse_bucketed_data(results, x_data="refcode", scoring_func="bleu1"):
    assert x_data in ["refcode", "gencode"]
    assert scoring_func in ["bleu1", "token_fraction"]
    refcode, refgen, gencode, _ = process_all_bleus(results, scoring_func=scoring_func)
    examples = results["results"]
    bucket_data = {}
    if scoring_func == "bleu1":
        bucket_labels = ["0", "0-36", "36-40", "40-45", ">45"]
        bucket_lower_limits = [-np.inf, 0, 36, 40, 45]
    elif scoring_func == "token_fraction":
        bucket_labels = ["0", "0-20", "20-40", "40-60", "60-80", "80-100"]
        bucket_lower_limits = [-np.inf, 0, 20, 40, 60, 80]
    for l, t in zip(bucket_labels, bucket_lower_limits):
        bucket_data[t] = {}
        bucket_data[t]["lower_limit"] = t
        bucket_data[t]["label"] = l
        bucket_data[t]["examples"] = []
    for example, rc, rg, gc in zip(examples, refcode, refgen, gencode):
        data = {}
        data["code"] = example["code"]
        data["ref"] = example["ref"]
        data["gen"] = example["gen"]
        data["refcode"] = rc
        data["refgen"] = rg
        data["gencode"] = gc
        if x_data == "refcode":
            limit_data = rc
        elif x_data == "gencode":
            limit_data = gc
        for limit in bucket_lower_limits[::-1]:
            if limit_data > limit:
                bucket_data[limit]["examples"].append(data)
                break
    for limit in bucket_data:
        bucket_data[limit]["size"] = len(bucket_data[limit]["examples"])
    return bucket_data


def return_hist(x, fixed_bins):
    xsize = len(x)
    if np.max(x) > fixed_bins[-1]:
        n_bins = fixed_bins + [np.max(x)]
    else:
        n_bins = fixed_bins +[fixed_bins[-1]+2]
    hist, bins = np.histogram(a=x, bins=n_bins, density=False)
    hist = (hist/xsize) * 100
    return hist


def get_bars(bucket_data, field, scoring_func="bleu1"):
    assert scoring_func in ["bleu1", "token_fraction"]
    # bins = [0, 1, 2, 3, 10, 15, 20, 25, 30, 50, 70, 1000]
    # These are labels for refcode and gencode (refgen  labels are along x-axis and were earlier defined)
    if scoring_func == "bleu1":
        bins = [0, 1, 30, 33, 36, 40, 45]
        bin_labels = ["0", "0-30", "30-33", "33-36", "36-40", "40-45", ">45"]
    elif scoring_func == "token_fraction":
        bins = [0, 1, 20, 40, 60, 80]
        bin_labels = ["0", "0-20", "20-40", "40-60", "60-80", "80-100"]
    bars = []
    overall_bar = []
    for limit, data in bucket_data.items():
        examples = data["examples"]
        elem = [example[field] for example in examples]
        bars.append(elem)
        overall_bar.extend(elem)
    
    bars.insert(0, overall_bar)
    bars = [return_hist(item, bins) for item in bars]

    bars = np.array(bars).T
    return bars, bin_labels, bins


def get_reverse_bars(bucket_data):
    bins = [0, 1, 10, 20, 30, 40]
    bin_labels = ['0', '0-10', '10-20', '20-30', '30-40', '>40']
    # labels = [f">{b}" for b in bins]
    bars = []
    overall_bar = []
    count = 0
    for limit, data in bucket_data.items():
        examples = data["examples"]
        elem = []
        for example in examples:
            ex = example['refgen']
            elem.append(ex)
        bars.append(elem)
        overall_bar.extend(elem)
        count += 1
    
    bars.insert(0, overall_bar)
    bars = [return_hist(item, bins) for item in bars]

    bars = np.array(bars).T
    return bars, bin_labels, bins


# x-axis is refgen and y-axis is refcode/gencode
def plot_stacked_barplot(args, bars, fixed_bins, x_labels, y_labels, category="reference", reverse=False, scoring_func="bleu1", variant_fname="old"):
    difficulty = args.diff
    assert category in ["reference", "generated"]
    assert scoring_func in ["bleu1", "token_fraction"]
    if scoring_func == "bleu1":
        scoring_str = "BLEU-1"
    elif scoring_func == "token_fraction":
        scoring_str = "t_copy"
    colors = sns.color_palette("crest", n_colors=len(fixed_bins)+1)
    fig, ax = plt.subplots()
    if reverse:
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    bottom = np.zeros(bars[0].shape)
    bars = np.flip(bars, 0)
    y_labels.reverse()
    colors.reverse()
    if difficulty:
        permutation = [0]
        for i in range(args.levels, 0, -1):
            permutation.append(i)
        idx = np.empty_like(permutation)
        idx[permutation] = np.arange(len(permutation))
        bars = bars[:, idx]  # return a rearranged copy
    for i, bar in enumerate(bars):
        lab = y_labels[i]
        if i == 0:
            ax.bar(x=x_labels, height=bar, color=colors[i], label=lab)
        else:
            ax.bar(x=x_labels, height=bar, bottom=bottom, color=colors[i], label=lab)
        bottom += bar
    if reverse:
        # ax.set_xlabel(f'{scoring_str} between {category.capitalize()} Desc. and Code')
        # ax.set_xlabel(r'Ranges of $p_{copy}$ Values ' + f'Between {category.capitalize()} Desc. and Code', fontsize=14)
        # ax.set_ylabel('BLEU-4 Scores between Reference and Generated Desc.')
        ax.set_ylabel('Dist. of BLEU-4 Scores', fontsize=14)
        reverse_str = "reverse_"
    else:
        ax.set_ylabel(f'{scoring_str} between {category.capitalize()} Desc. and Code')
        ax.set_xlabel('BLEU-4 Dist. between Reference and Generated Desc.')
        reverse_str = ""
    plt.ylim([0,100])
    plt.xticks(rotation =7)
    plt.gcf().subplots_adjust(bottom=0.15)
    fig.tight_layout()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height*0.9])

    # Put a legend to the right of the current axis
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
    legend_str = "BLEU-4" if reverse else scoring_str
    ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5),title=legend_str)
    # if reverse:
    #     # ax.set_title(f"Distribution of BLEU-4 for Code-Description {scoring_str} Ranges")
    #     plt.suptitle(r"Distribution of BLEU-4 By Range of $p_{copy}$ Values", ha='center', fontsize=16)
    #     # plt.title(r'By Range of $n_{copy}$ Values', ha='center', fontsize=16)
    # else:
    #     # ax.set_title(f"Distribution of Code-Description {scoring_str} For Different BLEU-4 Score Ranges")
    #     plt.suptitle(r'Distribution of Code-Description $n_{copy}$ For Different BLEU-4 Score Ranges')
    
    if difficulty:
        save_path = f"plots/stacked_bar/{variant_fname}/difficulty_{scoring_func}_{category}_code.pdf"
    else:
        save_path = f"plots/stacked_bar/{variant_fname}/flip_{reverse_str}{scoring_func}_{category}_code.pdf"
    print(save_path)
    plt.savefig(save_path)


def stacked_barplot(cfg, args, scoring_func):
    variant_fname = get_variant_name(cfg)
    os.makedirs(f"plots/stacked_bar/{variant_fname}", exist_ok=True)

    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    bucket_data = get_bucketed_data(results, scoring_func=scoring_func)
    
    bars_refcode, labels, bins = get_bars(bucket_data, "refcode", scoring_func=scoring_func)
    bars_gencode, _, _ = get_bars(bucket_data, "gencode", scoring_func=scoring_func)

    refgen_labels = [data['label'] for data in bucket_data.values()]
    refgen_labels.insert(0, "Overall")
    
    plot_stacked_barplot(args=args, bars=bars_refcode, fixed_bins=bins, x_labels=refgen_labels, y_labels=labels,
        category="reference", reverse=False, scoring_func=scoring_func, variant_fname=variant_fname)
    plot_stacked_barplot(args=args, bars=bars_gencode, fixed_bins=bins, x_labels=refgen_labels, y_labels=labels,
        category="generated", reverse=False, scoring_func=scoring_func, variant_fname=variant_fname)


# x-axis is refcode/gencode and y-axis is refgen
def reverse_stacked_barplot(cfg, args, scoring_func):
    variant_fname = get_variant_name(cfg)
    os.makedirs(f"plots/stacked_bar/{variant_fname}", exist_ok=True)

    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    refcode_bucket_data = get_reverse_bucketed_data(results, "refcode", scoring_func=scoring_func)
    gencode_bucket_data = get_reverse_bucketed_data(results, "gencode", scoring_func=scoring_func)
    
    bars_refcode, refgen_labels, bins = get_reverse_bars(refcode_bucket_data)
    bars_gencode, _, _ = get_reverse_bars(gencode_bucket_data)

    labels = [data['label'] for data in refcode_bucket_data.values()]
    labels.insert(0, "Overall")

    plot_stacked_barplot(args=args, bars=bars_refcode, fixed_bins=bins, x_labels=labels, y_labels=refgen_labels,
        category="reference", reverse=True, scoring_func=scoring_func, variant_fname=variant_fname)
    plot_stacked_barplot(args=args, bars=bars_gencode, fixed_bins=bins, x_labels=labels, y_labels=refgen_labels,
        category="generated", reverse=True, scoring_func=scoring_func, variant_fname=variant_fname)


# x-axis is refcode/gencode and y-axis is refgen
def get_difficulty_bucketed_data(args, results, x_data="refcode", scoring_func="bleu1"):
    assert x_data in ["refcode", "gencode"]
    assert scoring_func in ["bleu1", "token_fraction"]
    refcode, refgen, gencode, _ = process_all_bleus(results, scoring_func=scoring_func)
    examples = results["results"]
    bucket_data = {}
    bucket_labels = []
    for i in range(args.levels):
        bucket_labels.append(str(i+1))
    bucket_lower_limits = [-np.inf]
    vals = []
    for example, rc, rg, gc in zip(examples, refcode, refgen, gencode):
        if x_data == 'refcode':
            vals.append(rc)
        else:
            vals.append(rg)
    vals = sorted(vals)
    ln = len(vals)
    for i in range(1, args.levels):
        bucket_lower_limits.append(vals[int(ln*i/args.levels)])
    print(f"Thresholds for difficulty levels = {bucket_lower_limits}")

    for l, t in zip(bucket_labels, bucket_lower_limits):
        bucket_data[t] = {}
        bucket_data[t]["lower_limit"] = t
        bucket_data[t]["label"] = l
        bucket_data[t]["examples"] = []
    for example, rc, rg, gc in zip(examples, refcode, refgen, gencode):
        data = {}
        data["code"] = example["code"]
        data["ref"] = example["ref"]
        data["gen"] = example["gen"]
        data["refcode"] = rc
        data["refgen"] = rg
        data["gencode"] = gc
        if x_data == "refcode":
            limit_data = rc
        elif x_data == "gencode":
            limit_data = gc
        for limit in bucket_lower_limits[::-1]:
            if limit_data > limit:
                bucket_data[limit]["examples"].append(data)
                break
    for limit in bucket_data:
        bucket_data[limit]["size"] = len(bucket_data[limit]["examples"])
    return bucket_data


# x-axis is refcode/gencode and y-axis is refgen
def stacked_difficulty_barplot(cfg, args):
    scoring_func = args.scoring
    variant_fname = get_variant_name(cfg)
    os.makedirs(f"plots/stacked_bar/{variant_fname}", exist_ok=True)

    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    refcode_bucket_data = get_difficulty_bucketed_data(args, results, "refcode", scoring_func=scoring_func)
    gencode_bucket_data = get_difficulty_bucketed_data(args, results, "gencode", scoring_func=scoring_func)

    # print(refcode_bucket_data.keys())
    
    bars_refcode, refgen_labels, bins = get_reverse_bars(refcode_bucket_data)
    bars_gencode, _, _ = get_reverse_bars(gencode_bucket_data)

    labels = [data['label'] for data in refcode_bucket_data.values()]
    labels.insert(0, "Overall")

    plot_stacked_barplot(args=args, bars=bars_refcode, fixed_bins=bins, x_labels=labels, y_labels=refgen_labels,
        category="reference", reverse=True, scoring_func=scoring_func, variant_fname=variant_fname)
    plot_stacked_barplot(args=args, bars=bars_gencode, fixed_bins=bins, x_labels=labels, y_labels=refgen_labels,
        category="generated", reverse=True, scoring_func=scoring_func, variant_fname=variant_fname)


def sort_dict_by_value(d, reverse=False):
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))


def rank_sorted_dict_by_value(d):
    new_dict = {}
    for rank, key in enumerate(d, 1):
        if d[key] == 0.0:
            new_dict[key] = -1
        else:
            new_dict[key] = rank
    return new_dict


def aggregate_counts():
    columns = ["Total Frequency", "Token", "Counts Rank", "Probability Rank", "Counts", "Probability",
    "Negative Counts Rank", "Negative Probability Rank", "Negative Counts", "Negative Probability"]
    ref_data = {}
    gen_data = {}
    for col in columns:
        ref_data[col] = []
        gen_data[col] = []

    ref_counts_overlap = json.load(open('counts/ref_counts_overlap.json'))
    ref_counts_non_overlap = json.load(open('counts/ref_counts_non_overlap.json'))
    ref_counts = json.load(open('counts/ref_counts.json'))
    gen_counts_overlap = json.load(open('counts/gen_counts_overlap.json'))
    gen_counts_non_overlap = json.load(open('counts/gen_counts_non_overlap.json'))
    gen_counts = json.load(open('counts/gen_counts.json'))

    ref_probs_overlap = json.load(open('counts/probs/ref_counts_overlap.json'))
    ref_probs_non_overlap = json.load(open('counts/probs/ref_counts_non_overlap.json'))
    gen_probs_overlap = json.load(open('counts/probs/gen_counts_overlap.json'))
    gen_probs_non_overlap = json.load(open('counts/probs/gen_counts_non_overlap.json'))

    ref_rankprobs_overlap = rank_sorted_dict_by_value(ref_probs_overlap)
    ref_rankprobs_non_overlap = rank_sorted_dict_by_value(ref_probs_non_overlap)
    gen_rankprobs_overlap = rank_sorted_dict_by_value(gen_probs_overlap)
    gen_rankprobs_non_overlap = rank_sorted_dict_by_value(gen_probs_non_overlap)

    ref_rankcounts_overlap = rank_sorted_dict_by_value(ref_counts_overlap)
    gen_rankcounts_overlap = rank_sorted_dict_by_value(gen_counts_overlap)
    ref_rankcounts_non_overlap = rank_sorted_dict_by_value(ref_counts_non_overlap)
    gen_rankcounts_non_overlap = rank_sorted_dict_by_value(gen_counts_non_overlap)

    for tok in ref_counts:
        ref_data["Total Frequency"].append(ref_counts[tok])
        ref_data["Token"].append(tok)
        ref_data["Counts Rank"].append(ref_rankcounts_overlap.get(tok, -1))
        ref_data["Probability Rank"].append(ref_rankprobs_overlap[tok])
        ref_data["Counts"].append(ref_counts_overlap.get(tok, 0))
        ref_data["Probability"].append(ref_probs_overlap[tok])
        ref_data["Negative Counts Rank"].append(ref_rankcounts_non_overlap.get(tok, -1))
        ref_data["Negative Probability Rank"].append(ref_rankprobs_non_overlap[tok])
        ref_data["Negative Counts"].append(ref_counts_non_overlap.get(tok, 0))
        ref_data["Negative Probability"].append(ref_probs_non_overlap[tok])

    for tok in gen_counts:
        gen_data["Total Frequency"].append(gen_counts[tok])
        gen_data["Token"].append(tok)
        gen_data["Counts Rank"].append(gen_rankcounts_overlap.get(tok, -1))
        gen_data["Probability Rank"].append(gen_rankprobs_overlap[tok])
        gen_data["Counts"].append(gen_counts_overlap.get(tok, 0))
        gen_data["Probability"].append(gen_probs_overlap[tok])
        gen_data["Negative Counts Rank"].append(gen_rankcounts_non_overlap.get(tok, -1))
        gen_data["Negative Probability Rank"].append(gen_rankprobs_non_overlap[tok])
        gen_data["Negative Counts"].append(gen_counts_non_overlap.get(tok, 0))
        gen_data["Negative Probability"].append(gen_probs_non_overlap[tok])

    ref_df = pd.DataFrame(data=ref_data)
    gen_df = pd.DataFrame(data=gen_data)

    ref_df.to_csv(path_or_buf="counts/ref_df.csv")
    gen_df.to_csv(path_or_buf="counts/gen_df.csv")

    code_counts = json.load(open('counts/code_counts.json'))
    code_ref_counts_overlap = json.load(open('counts/code_ref_counts_overlap.json'))
    code_ref_counts_non_overlap = json.load(open('counts/code_ref_counts_non_overlap.json'))
    code_gen_counts_overlap = json.load(open('counts/code_gen_counts_overlap.json'))
    code_gen_counts_non_overlap = json.load(open('counts/code_gen_counts_non_overlap.json'))

    code_ref_probs_overlap = json.load(open('counts/probs/code_ref_counts_overlap.json'))
    code_ref_probs_non_overlap = json.load(open('counts/probs/code_ref_counts_non_overlap.json'))
    code_gen_probs_overlap = json.load(open('counts/probs/code_gen_counts_overlap.json'))
    code_gen_probs_non_overlap = json.load(open('counts/probs/code_gen_counts_non_overlap.json'))

    code_ref_rankprobs_overlap = rank_sorted_dict_by_value(code_ref_probs_overlap)
    code_ref_rankprobs_non_overlap = rank_sorted_dict_by_value(code_ref_probs_non_overlap)
    code_gen_rankprobs_overlap = rank_sorted_dict_by_value(code_gen_probs_overlap)
    code_gen_rankprobs_non_overlap = rank_sorted_dict_by_value(code_gen_probs_non_overlap)

    code_ref_rankcounts_overlap = rank_sorted_dict_by_value(code_ref_counts_overlap)
    code_gen_rankcounts_overlap = rank_sorted_dict_by_value(code_gen_counts_overlap)
    code_ref_rankcounts_non_overlap = rank_sorted_dict_by_value(code_ref_counts_non_overlap)
    code_gen_rankcounts_non_overlap = rank_sorted_dict_by_value(code_gen_counts_non_overlap)

    code_columns = ["Total Frequency", "Token", "Ref Counts Rank", "Ref Probability Rank", "Ref Counts",
    "Ref Probability", "Ref Negative Counts Rank", "Ref Negative Probability Rank", "Ref Negative Counts",
    "Ref Negative Probability", "Gen Counts Rank", "Gen Probability Rank", "Gen Counts",
    "Gen Probability", "Gen Negative Counts Rank", "Gen Negative Probability Rank", "Gen Negative Counts",
    "Gen Negative Probability"]
    code_data = {}
    for col in code_columns:
        code_data[col] = []

    for tok in code_counts:
        code_data["Total Frequency"].append(code_counts[tok])
        code_data["Token"].append(tok)
        code_data["Ref Counts Rank"].append(code_ref_rankcounts_overlap.get(tok, -1))
        code_data["Ref Probability Rank"].append(code_ref_rankprobs_overlap[tok])
        code_data["Ref Counts"].append(code_ref_counts_overlap.get(tok, 0))
        code_data["Ref Probability"].append(code_ref_probs_overlap[tok])
        code_data["Ref Negative Counts Rank"].append(code_ref_rankcounts_non_overlap.get(tok, -1))
        code_data["Ref Negative Probability Rank"].append(code_ref_rankprobs_non_overlap[tok])
        code_data["Ref Negative Counts"].append(code_ref_counts_non_overlap.get(tok, 0))
        code_data["Ref Negative Probability"].append(code_ref_probs_non_overlap[tok])

        code_data["Gen Counts Rank"].append(code_gen_rankcounts_overlap.get(tok, -1))
        code_data["Gen Probability Rank"].append(code_gen_rankprobs_overlap[tok])
        code_data["Gen Counts"].append(code_gen_counts_overlap.get(tok, 0))
        code_data["Gen Probability"].append(code_gen_probs_overlap[tok])
        code_data["Gen Negative Counts Rank"].append(code_gen_rankcounts_non_overlap.get(tok, -1))
        code_data["Gen Negative Probability Rank"].append(code_gen_rankprobs_non_overlap[tok])
        code_data["Gen Negative Counts"].append(code_gen_counts_non_overlap.get(tok, 0))
        code_data["Gen Negative Probability"].append(code_gen_probs_non_overlap[tok])

    code_df = pd.DataFrame(data=code_data)
    code_df.to_csv(path_or_buf="counts/code_df.csv")


def count_overlap(cfg):
    pretrained_model="Salesforce/codet5-base-multi-sum"
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    ref_counts_overlap = defaultdict(int)
    ref_counts_non_overlap = defaultdict(int)
    ref_counts = defaultdict(int)
    gen_counts_overlap = defaultdict(int)
    gen_counts_non_overlap = defaultdict(int)
    gen_counts = defaultdict(int)
    code_ref_counts_overlap = defaultdict(int)
    code_ref_counts_non_overlap = defaultdict(int)
    code_gen_counts_overlap = defaultdict(int)
    code_gen_counts_non_overlap = defaultdict(int)
    code_counts = defaultdict(int)
    for data in tqdm(results["results"]):
        c_tokens = get_tokens(tokenizer, data['code'])
        r_tokens = get_tokens(tokenizer, data['ref'])
        g_tokens = get_tokens(tokenizer, data['gen'])
        for r in r_tokens:
            if r in c_tokens:
                ref_counts_overlap[r] += 1
            else:
                ref_counts_non_overlap[r] += 1
            ref_counts[r] += 1
        for g in g_tokens:
            if g in c_tokens:
                gen_counts_overlap[g] += 1
            else:
                gen_counts_non_overlap[g] += 1
            gen_counts[g] += 1
        for c in c_tokens:
            if c in r_tokens:
                code_ref_counts_overlap[c] += 1
            else:
                code_ref_counts_non_overlap[c] += 1
            if c in g_tokens:
                code_gen_counts_overlap[c] += 1
            else:
                code_gen_counts_non_overlap[c] += 1
            code_counts[c] += 1
    
    ref_counts_overlap = sort_dict_by_value(ref_counts_overlap, reverse=True)
    ref_counts_non_overlap = sort_dict_by_value(ref_counts_non_overlap, reverse=True)
    ref_counts = sort_dict_by_value(ref_counts, reverse=True)
    gen_counts_overlap = sort_dict_by_value(gen_counts_overlap, reverse=True)
    gen_counts_non_overlap = sort_dict_by_value(gen_counts_non_overlap, reverse=True)
    gen_counts = sort_dict_by_value(gen_counts, reverse=True)
    code_counts = sort_dict_by_value(code_counts, reverse=True)
    code_ref_counts_overlap = sort_dict_by_value(code_ref_counts_overlap, reverse=True)
    code_ref_counts_non_overlap = sort_dict_by_value(code_ref_counts_non_overlap, reverse=True)
    code_gen_counts_overlap = sort_dict_by_value(code_gen_counts_overlap, reverse=True)
    code_gen_counts_non_overlap = sort_dict_by_value(code_gen_counts_non_overlap, reverse=True)

    json.dump(ref_counts_overlap, open('counts/ref_counts_overlap.json', 'w'), indent=2)
    json.dump(ref_counts_non_overlap, open('counts/ref_counts_non_overlap.json', 'w'), indent=2)
    json.dump(ref_counts, open('counts/ref_counts.json', 'w'), indent=2)
    json.dump(gen_counts_overlap, open('counts/gen_counts_overlap.json', 'w'), indent=2)
    json.dump(gen_counts_non_overlap, open('counts/gen_counts_non_overlap.json', 'w'), indent=2)
    json.dump(gen_counts, open('counts/gen_counts.json', 'w'), indent=2)
    json.dump(code_counts, open('counts/code_counts.json', 'w'), indent=2)
    json.dump(code_ref_counts_overlap, open('counts/code_ref_counts_overlap.json', 'w'), indent=2)
    json.dump(code_ref_counts_non_overlap, open('counts/code_ref_counts_non_overlap.json', 'w'), indent=2)
    json.dump(code_gen_counts_overlap, open('counts/code_gen_counts_overlap.json', 'w'), indent=2)
    json.dump(code_gen_counts_non_overlap, open('counts/code_gen_counts_non_overlap.json', 'w'), indent=2)

    for tok, count in ref_counts.items():
        ref_counts_overlap[tok] = ref_counts_overlap.get(tok, 0) / count
        ref_counts_non_overlap[tok] = ref_counts_non_overlap.get(tok, 0) / count
    for tok, count in gen_counts.items():
        gen_counts_overlap[tok] = gen_counts_overlap.get(tok, 0) / count
        gen_counts_non_overlap[tok] = gen_counts_non_overlap.get(tok, 0) / count

    for tok, count in code_counts.items():
        code_ref_counts_overlap[tok] = code_ref_counts_overlap.get(tok, 0) / count
        code_ref_counts_non_overlap[tok] = code_ref_counts_non_overlap.get(tok, 0) / count
        code_gen_counts_overlap[tok] = code_gen_counts_overlap.get(tok, 0) / count
        code_gen_counts_non_overlap[tok] = code_gen_counts_non_overlap.get(tok, 0) / count

    ref_counts_overlap = sort_dict_by_value(ref_counts_overlap, reverse=True)
    ref_counts_non_overlap = sort_dict_by_value(ref_counts_non_overlap, reverse=True)
    ref_counts = sort_dict_by_value(ref_counts, reverse=True)
    gen_counts_overlap = sort_dict_by_value(gen_counts_overlap, reverse=True)
    gen_counts_non_overlap = sort_dict_by_value(gen_counts_non_overlap, reverse=True)
    gen_counts = sort_dict_by_value(gen_counts, reverse=True)
    code_counts = sort_dict_by_value(code_counts, reverse=True)
    code_ref_counts_overlap = sort_dict_by_value(code_ref_counts_overlap, reverse=True)
    code_ref_counts_non_overlap = sort_dict_by_value(code_ref_counts_non_overlap, reverse=True)
    code_gen_counts_overlap = sort_dict_by_value(code_gen_counts_overlap, reverse=True)
    code_gen_counts_non_overlap = sort_dict_by_value(code_gen_counts_non_overlap, reverse=True)

    json.dump(ref_counts_overlap, open('counts/probs/ref_counts_overlap.json', 'w'), indent=2)
    json.dump(ref_counts_non_overlap, open('counts/probs/ref_counts_non_overlap.json', 'w'), indent=2)
    json.dump(gen_counts_overlap, open('counts/probs/gen_counts_overlap.json', 'w'), indent=2)
    json.dump(gen_counts_non_overlap, open('counts/probs/gen_counts_non_overlap.json', 'w'), indent=2)
    json.dump(code_ref_counts_overlap, open('counts/probs/code_ref_counts_overlap.json', 'w'), indent=2)
    json.dump(code_ref_counts_non_overlap, open('counts/probs/code_ref_counts_non_overlap.json', 'w'), indent=2)
    json.dump(code_gen_counts_overlap, open('counts/probs/code_gen_counts_overlap.json', 'w'), indent=2)
    json.dump(code_gen_counts_non_overlap, open('counts/probs/code_gen_counts_non_overlap.json', 'w'), indent=2)


def plot_variant_statistics(cfg):
    data = pd.DataFrame(columns=["Variant", "refcode_mean", "refcode_median", "gencode_mean", "gencode_median"])
    all_refcodes = []
    all_gencodes = []
    for variant in variants:
        print(variant)
        if variant == (False, False, True, False):
            continue
        if variant[2]:
            variant = (variant[0], variant[1], False, variant[3])
        cfg["remove_func"], cfg["only_func"], cfg["remove_comments"], cfg["adversarial_func"] = variant
        if cfg["remove_func"] == KEYWORD:
            cfg["remove_func"] = False
            cfg["remove_keyword"] = True
        else:
            cfg["remove_keyword"] = False
        refcode_mean, refcode_median, gencode_mean, gencode_median, refcode, gencode = simple_histograms(cfg, args.scoring, only_stats=True)
        all_refcodes.append(refcode)
        all_gencodes.append(gencode)
        data.loc[len(data.index)] = [variant_mapping[str(variant)], refcode_mean, refcode_median, gencode_mean, gencode_median] 
    print(data)
    
    refcode_means = data['refcode_mean'].tolist()
    gencode_means = data['gencode_mean'].tolist()
    variant_names = data['Variant'].tolist()

    new_variant_names = []
    for varn in variant_names:
        if varn=="Original Function Names (without Comments)":
            newval = varn
        else:
            newval = varn.replace(' (without Comments)', '')
        new_variant_names.append(newval.replace(' ', '\n'))
    variant_names = new_variant_names
    
    # dims = set_size(WIDTH)
    # dims = (dims[0], 5.0)
    # print(dims)
    # fig, ax = plt.subplots(figsize=dims)
    fig, ax = plt.subplots()
    # ax.plot(variant_names, gencode_means, linestyle='--', marker='o', label="Generated", color="blue")
    # ax.plot(variant_names, refcode_means, linestyle='--', marker='o', label="Reference", color="orange")
    # ax.yaxis.set_major_formatter(mtick.PercentFormatter())

    X_axis = np.arange(1, len(variant_names)+1)
    colors = sns.color_palette("colorblind", n_colors=5)
    # ax.bar(X_axis+0.2, gencode_means, width=0.4, label="Generated", color="midnightblue")
    # ax.bar(X_axis-0.2, refcode_means, width=0.4, label="Reference", color="chocolate")

    # ax.bar(variant_names, gen_means, width=0.4, color="midnightblue")

    bp1 = ax.boxplot(all_refcodes, widths=0.3, positions=X_axis-0.16, notch=True,
        showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[1]),
        medianprops={"color": "black", "linewidth": 2, "solid_capstyle": "butt"})
    bp2 = ax.boxplot(all_gencodes, widths=0.3, positions=X_axis+0.16, notch=True,
        showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[0]),
        medianprops={"color": "black", "linewidth": 2, "solid_capstyle": "butt"})
    

    plt.xticks(X_axis, variant_names)

    ax.set_ylabel(r'Distribution of $p_{copy}$', fontsize=14)
    ax.set_xlabel("Variants", fontsize=14)
    # plt.suptitle(r"Boxplot of $p_{copy}$ For Each Variant", ha='center', fontsize=20)
    # Put a legend to the right of the current axis
    # handles, labels = ax.get_legend_handles_labels()
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Reference', 'Generated'], loc='upper right')
    # ax.legend(handles[::-1], labels[::-1], loc='upper right',title="Legend")
    plt.xticks(rotation = 10)
    plt.tight_layout()
    plt.savefig(f"plots/token_overlap_variant.pdf")
    plt.close()


def plot_sequence_lengths(cfg):
    df = pd.DataFrame(columns=["Variant", "ref_mean", "ref_median", "gen_mean", "gen_median"])
    all_gen_lens = []
    for variant in variants:
        cfg["remove_func"], cfg["only_func"], cfg["remove_comments"], cfg["adversarial_func"] = variant
        if cfg["remove_func"] == KEYWORD:
            cfg["remove_func"] = False
            cfg["remove_keyword"] = True
        else:
            cfg["remove_keyword"] = False
        result_path = get_result_path(cfg)
        results = json.load(open(result_path))
        pretrained_model="Salesforce/codet5-base-multi-sum"
        tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
        ref_lens = []
        gen_lens = []
        for data in tqdm(results['results']):
            # code = get_tokens(tokenizer, data['code'])
            ref = get_tokens(tokenizer, data['ref'])
            gen = get_tokens(tokenizer, data['gen'])
            ref_lens.append(len(ref))
            gen_lens.append(len(gen))
        ref_mean = np.mean(ref_lens)
        ref_median = np.median(ref_lens)
        gen_mean = np.mean(gen_lens)
        all_gen_lens.append(gen_lens)
        gen_median = np.median(gen_lens)
        df.loc[len(df.index)] = [variant_mapping[str(variant)], ref_mean, ref_median, gen_mean, gen_median] 
        print(variant)
        print(ref_mean)
        print(ref_median)
        print(gen_mean)
        print(gen_median)
        exit()
        
    ref_means = df['ref_mean'].tolist()
    gen_means = df['gen_mean'].tolist()
    variant_names = df['Variant'].tolist()

    new_variant_names = []
    for varn in variant_names:
        if varn=="Original Function Names (without Comments)":
            newval = varn
        else:
            newval = varn.replace(' (without Comments)', '')
        new_variant_names.append(newval.replace(' ', '\n'))
    variant_names = new_variant_names

    # fig, ax = plt.subplots(figsize=set_size(WIDTH))
    fig, ax = plt.subplots()
    # ax.plot(variant_names, gen_means, linestyle='--', marker='o')
    # ax.bar(variant_names, gen_means, width=0.4, color="midnightblue")

    colors = sns.color_palette("colorblind", n_colors=2)
    ax.boxplot(all_gen_lens, notch=True, showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[0]), medianprops={"color": "black", "linewidth": 2, "solid_capstyle": "butt"})
    X_axis = list(range(1, len(variant_names) + 1))
    plt.xticks(X_axis, variant_names, rotation = 10)

    ax.set_ylabel('Distribution of Sequence Lengths', fontsize=14)
    ax.set_xlabel("Variants", fontsize=14)
    # plt.suptitle("Boxplot of Generated Desc. Sequence Lengths", ha='center', fontsize=20)
    # plt.title("For Each Variant", ha="center", fontsize=16)
    # Put a legend to the right of the current axis
    handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title='Line', loc='upper left')
    # ax.legend(handles[::-1], labels[::-1], loc='center left', bbox_to_anchor=(1, 0.5),title="Reference or Generated")
    # ax.legend(handles[::-1], labels[::-1], loc='lower left',title="Legend")
    # plt.xticks(rotation = 10)
    plt.tight_layout()
    plt.savefig(f"plots/sequence_length_variant.pdf")
    plt.close()

# Returns indices in dataset that have comments in code
def get_comments_idxs(cfg):
    if cfg["remove_comments"]:
        originally_remove = True
    else:
        originally_remove = False
    cfg["remove_comments"] = False
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    with_comments_dataset = results['results']
    cfg["remove_comments"] = True
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    comments_idxs = []
    without_comments_dataset = results['results']
    for i, (w, wo) in enumerate(zip(with_comments_dataset, without_comments_dataset)):
        if w['code'] != wo['code']:
            comments_idxs.append(i)
    # print(f"Number of examples with comments = {len(comments_idxs)}")
    # print(f"Total number of examples = {len(with_comments_dataset)}")
    if originally_remove:
        cfg["remove_comments"] = True
    else:
        cfg["remove_comments"] = False
    return comments_idxs


def find_comments(args, cfg):
    result_path = get_result_path(cfg)
    results = json.load(open(result_path))
    pretrained_model="Salesforce/codet5-base-multi-sum"
    tokenizer = RobertaTokenizer.from_pretrained(pretrained_model, do_lower_case=True)
    comments_idxs = get_comments_idxs(cfg)
    with_comments = []
    without_comments = []
    with_comments_score = 0.0
    without_comments_score = 0.0
    total_score = 0.0
    print(results['bleu_score'])
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    for i, data in enumerate(tqdm(results['results'])):
        code = data['code']
        r_tokens = tokens = tokenizer.tokenize(data['ref'].lower())
        g_tokens = tokens = tokenizer.tokenize(data['gen'].lower())
        example = {'code': code, 'ref': [r_tokens], 'gen': g_tokens}
        score = nltk.translate.bleu_score.sentence_bleu(example['ref'], example['gen'], smoothing_function=chencherry.method2)
        if i in comments_idxs:
            with_comments.append(example)
            with_comments_score += score
        else:
            without_comments.append(example)
            without_comments_score += score
        total_score += score

    with_comments_score = 100.0 * with_comments_score / (len(with_comments))
    without_comments_score = 100.0 * without_comments_score / (len(without_comments))
    total_score = 100.0 * total_score / (len(results['results']))
    print(f"BLEU Score for examples with comments = {with_comments_score}")
    print(f"BLEU Score for examples without comments = {without_comments_score}")
    print(f"Overall BLEU Score = {total_score}")
    print(f"Total examples = {len(results['results'])}")
    print(f"Percentage of examples with comments = {len(with_comments) / len(results['results'])}")


def main(args, cfg):
    if args.task == 'translate':
        print("Running translate")
        translate(args, cfg)
        save_bleu(cfg)
    if args.task == 'score':
        print("Saving BLEU")
        save_bleu(cfg)
    if args.task == 'view':
        print("Viewing Results")
        view_results(cfg)
    if args.task == 'analyze':
        print("Analyzing and Saving Results")
        save_all_bleu(cfg)
    if args.task == 'hist':
        print("Plotting Histogram")
        simple_histograms(cfg, args.scoring)
    if args.task == 'stacked' and not args.diff:
        print("Plotting Stacked Bar Plot")
        if args.nrev:
            stacked_barplot(cfg, args, args.scoring)
        else:
            reverse_stacked_barplot(cfg, args, args.scoring)
    if args.task == 'stacked' and args.diff:
        print("Plotting Stacked Bar Plot")
        stacked_difficulty_barplot(cfg, args)
    if args.task == 'count':
        print("Counting...")
        count_overlap(cfg)
        aggregate_counts()
    if args.task == 'stats':
        assert args.allv is not True
        print("Computing and Plotting Stats")
        plot_variant_statistics(cfg)
    if args.task == 'seqlens':
        plot_sequence_lengths(cfg)
    if args.task == 'comments':
        find_comments(args, cfg)


if __name__=="__main__":
    tasks = ['translate', 'score', 'view', 'analyze', 'hist', 'stacked', 'count', 'stats', 'seqlens', 'comments']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/config.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=1, type=int,
        help='seed for random number generation')
    parser.add_argument(
        '--task', default="translate", type=str,
        choices=tasks,
        help='Task to perform')
    parser.add_argument(
        '--scoring', default="token_fraction", type=str,
        help='Scoring Function - bleu1 or token_fraction')
    parser.add_argument('--allv', action='store_true',
        help='Flag to check if processing for all variants')
    parser.add_argument('--nrev', action='store_true',
        help='Flag to check if stacked bar plot is reverse or not')
    parser.add_argument('--diff', action='store_true',
        help='Flag to check if using difficulty levels instead of buckets')
    parser.add_argument('--levels', default=6, type=int,
        help='Number of difficulty levels')
    args = parser.parse_args()
    # Set random seed for reproducibility
    set_seed(args)
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    if args.allv:
        for variant in variants:
            cfg["remove_func"], cfg["only_func"], cfg["remove_comments"], cfg["adversarial_func"] = variant
            if cfg["remove_func"] == KEYWORD:
                cfg["remove_func"] = False
                cfg["remove_keyword"] = True
            else:
                cfg["remove_keyword"] = False
            main(args, cfg)
    else:
        main(args, cfg)