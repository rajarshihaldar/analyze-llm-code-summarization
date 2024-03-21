"""Dataset for loading raw JSON Lines data for CodeSearchNet."""

import json
import os
import random
import keyword
import operator

import numpy as np
from tqdm import tqdm, trange

from llm_datasets import BaseDataset

from llm_datasets.transform_dataset import modify_func_names, obscure, remove_comments_from_code, extract_comments_from_code

from utils import OPERATORS, PUNCTUATIONS

def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


class CodeXGlue(BaseDataset):
    """CodeSearchNet dataset."""

    num_files = {
        'train': 1,
        'valid': 1,
        'test': 1,
    }

    def _load(self):
        """Load a dataset into memory."""
        data = []
        # Loop over all files in dataset
        fname = '{}.jsonl'.format(self.dataset)
        func_names = []
        with open(os.path.join(self.root, fname)) as f:
            # Loop over code/desc pairs in file
            for line in f:
                # Parse JSON string
                obj = json.loads(line)

                # Truncate description to first paragraph
                desc = obj['docstring'].split('\n\n')[0]
                if self.use_code_tokens:
                    code = obj['code']
                else:
                    code = obj['code_tokens']
                if self.adversarial_func and (not self.use_code_tokens):
                    index = code.index('def')
                    func_name = code[index+1]
                    func_names.append(func_name)
                if self.adversarial_func and self.use_code_tokens:
                    left = 4
                    right = code.find('(')
                    func = code[left:right]
                    func_names.append(func)
                if self.use_code_tokens and self.remove_func and (not self.adversarial_func):
                    code = modify_func_names(code, self.use_code_tokens)
                    if self.remove_comments:
                        code = remove_comments_from_code(code)
                if self.use_code_tokens and self.only_func and (not self.adversarial_func):
                    body = code.find('"""', code.find('"""')+1)
                    code = code[:body+3]
                    if self.remove_comments:
                        code = remove_comments_from_code(code)
                if (not self.use_code_tokens) and (not self.adversarial_func) and (not self.original_tokens):
                    flag = True
                    if self.remove_keyword:
                        code = [token for token in code if token not in keyword.kwlist]
                        code = [token for token in code if token not in OPERATORS]
                        code = [token for token in code if token not in PUNCTUATIONS]
                        
                    if (not self.only_func) and self.remove_comments:
                        code = remove_comments_from_code(code)
                    elif self.only_func and (not self.remove_comments):
                        comments = extract_comments_from_code(code)
                        try:
                            index = code.index(':')
                            code = code[:index]
                            # index = code.index('def')
                            # code = code[index+1]
                            # code = code.split('_')
                            code = ' '.join(code) + ' ' + ' '.join(comments)
                            flag = False
                        except (ValueError, IndexError):
                            code = comments
                            pass
                    elif self.only_func and self.remove_comments:
                        try:
                            index = code.index(':')
                            code = code[:index]
                            # index = code.index('def')
                            # code = code[index+1]
                            # code = code.split('_')
                            code = ' '.join(code)
                            flag = False
                        except (ValueError, IndexError):
                            pass
                    if self.remove_func:
                        try:
                            index = code.index('def')
                            code[index + 1] = obscure(code[index + 1])
                        except (ValueError, IndexError):
                            pass
                    if flag:
                        code = ' '.join([format_str(token) for token in code])
                data.append((code, desc))
        if self.adversarial_func and (not self.use_code_tokens):
            new_data = []
            for d in tqdm(data):
                func_name = random.sample(func_names, 1)[0]
                code = d[0]
                desc = d[1]
                if self.remove_comments:
                    code = remove_comments_from_code(code)
                index = code.index('def')
                code[index+1] = func_name
                code = ' '.join([format_str(token) for token in code])
                new_data.append((code, desc))
            data = new_data
        if self.adversarial_func and self.use_code_tokens:
            new_data = []
            for d in data:
                func_name = random.sample(func_names, 1)[0]
                code = d[0]
                desc = d[1]
                left = 4
                right = code.find('(')
                func = code[left:right]
                code = code.replace(func, func_name)
                if self.remove_comments:
                    code = remove_comments_from_code(code)
                new_data.append((code, desc))
            data = new_data

        # self.data = np.array(data, dtype=np.object
        self.data = np.array(data)
