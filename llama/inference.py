import os
import re
import time
import argparse
import random
import pickle
from collections import defaultdict
import pprint

import sys
import inspect
from tqdm import tqdm
import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from llm_datasets.codexglue import CodeXGlue
from utils import set_seed
from create_dataset import get_codexglue


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    '-c', '--config', default='configs/1.yaml',
    help='YAML file containing configuration parameters', metavar='FILE')
parser.add_argument(
    '-s', '--seed', default=42, type=int,
    help='seed for random number generation')
parser.add_argument(
    '--mtype', default='codellama', type=str,
    choices=['llama2', 'codellama'],
    help='Number of parameters in Llama2 to load')
parser.add_argument(
    '--parameters', default='7b', type=str,
    choices=['7b', '13b', '70b'],
    help='Number of parameters in Llama2 to load')
args = parser.parse_args()
# Set random seed for reproducibility
set_seed(args)

model_size = args.parameters
load_quantized = True

cache_dir = './'
if args.mtype == 'llama2':
    pretrained_model = f'meta-llama/Llama-2-{model_size}-chat-hf'
else:
    pretrained_model = f'codellama/CodeLlama-{model_size}-Instruct-hf'
tokenizer = AutoTokenizer.from_pretrained(pretrained_model,
    cache_dir=cache_dir)
model =  AutoModelForCausalLM.from_pretrained(pretrained_model,
    load_in_4bit=load_quantized,
    cache_dir=cache_dir)

llama_template = f"""
<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

[INSERT_PROMPT] [/INST]
"""


def generate_desc(prompt):
    input_prompt = llama_template.replace('[INSERT_PROMPT]', prompt)
    x = tokenizer(prompt, return_tensors='pt').to('cuda')
    y = model.generate(**x, max_new_tokens=1000, do_sample=True, temperature=0.1)
    y = y[0]
    y = tokenizer.decode(
        y, skip_special_tokens=True, spaces_between_special_tokens=False
            )
    y = y.split('Documentation:')[-1].strip()
    y = y.split('\n')[0]
    return y


def generate_random_indices(random_samples, dataset):
    indices = list(range(len(dataset)))
    indices = random.sample(indices, random_samples)
    pickle.dump(indices, open('random_indices.pkl', 'wb'))


def main(cfg, args):
    with open('prompt.txt') as f:
        instruction = f.read()
    remove_func = cfg.get('remove_func', False)
    only_func = cfg.get('only_func', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    remove_comments = True
    data_split = 'test'
    root = '../'
    if remove_keyword:
        use_code_tokens = False
    else:
        use_code_tokens = True
    dataset = CodeXGlue(root, dataset=data_split, remove_func=remove_func,
        only_func=only_func, remove_comments=remove_comments, adversarial_func=adversarial_func,
        remove_keyword=remove_keyword, use_code_tokens=use_code_tokens)
    
    use_random_samples = False
    random_samples = 1000

    if use_random_samples:
        # generate_random_indices(random_samples, dataset)
        # exit()
        indices = pickle.load(open('random_indices.pkl', 'rb'))
        random_indices = defaultdict(bool)
        for ind in indices:
            random_indices[ind] = True

    if use_random_samples:
        new_dataset = []
        for i, item in enumerate(dataset):
            if not random_indices[i]:
                continue
            code, desc = item
            new_dataset.append((code, desc))
        dataset = new_dataset

    fname = f'results/original_{args.mtype}_{model_size}.txt'
    if remove_func:
        fname = f'results/obfusc_{args.mtype}_{model_size}.txt'
    if adversarial_func:
        fname = f'results/advers_{args.mtype}_{model_size}.txt'
    if remove_keyword:
        fname = f'results/remove_keyword_{args.mtype}_{model_size}.txt'
    if only_func:
        fname = f'results/no_body_{args.mtype}_{model_size}.txt'
    with open(fname, 'w') as f:
        pass

    for code, desc in tqdm(dataset):
        doc_left = code.find('"""')
        doc_right = code.find('"""', code.find('"""')+1)
        doc = code[doc_left:doc_right+3]
        code = code.replace(doc, '')
        prompt = instruction + code + '\n' + "Documentation:"
        generation = generate_desc(prompt)
        if generation is None:
            generation = 'None'
        with open(fname, 'a') as f:
            f.write("\n<Example Start>\n")
            f.write("\n<Code>\n")
            f.write(code)
            f.write("\n<Reference Desc>\n")
            f.write(desc)
            f.write("\n<Generated Desc>\n")
            f.write(generation)
            f.write('\n')


if __name__=="__main__":
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)
