import os
import re
import time
import argparse
import random
import pickle
from collections import defaultdict
import pprint
import google.generativeai as palm

import sys
import inspect
from tqdm import tqdm
import yaml

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from datasets.codexglue import CodeXGlue
from utils import set_seed
from create_dataset import get_codexglue


palm.configure(api_key=os.getenv('PALM_KEY'))
models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name


def generate_desc(prompt, max_output_tokens):
    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0,
        # The maximum length of the response
        max_output_tokens=max_output_tokens,
    )

    return completion.result


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
    max_output_tokens = cfg.get('max_output_tokens', 128)
    time_delay = cfg.get('time_delay', 2.01)
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

    fname = 'results/palm.txt'
    if remove_func:
        fname = 'results/palm_obfusc.txt'
    if adversarial_func:
        fname = 'results/palm_advers.txt'
    if remove_keyword:
        fname = 'results/palm_remove_keyword.txt'
    if only_func:
        fname = 'results/palm_no_body.txt'
    with open(fname, 'w') as f:
        pass

    for code, desc in tqdm(dataset):
        doc_left = code.find('"""')
        doc_right = code.find('"""', code.find('"""')+1)
        doc = code[doc_left:doc_right+3]
        code = code.replace(doc, '')
        prompt = instruction + code + '\n' + "Documentation:"
        generation = generate_desc(prompt, max_output_tokens)
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
        time.sleep(time_delay)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/1.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=42, type=int,
        help='seed for random number generation')
    args = parser.parse_args()
    # Set random seed for reproducibility
    set_seed(args)
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)
