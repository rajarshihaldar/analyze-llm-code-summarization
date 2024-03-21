"""Dataset for loading raw JSON data for CoNaLa,
the Code/Natural Language Challenge.
"""

import json
import os

import numpy as np

from codet5_datasets import BaseDataset
from codet5_datasets.transform_dataset import modify_func_names, obscure, remove_comments_from_code, extract_comments_from_code, tokenize_code


def format_str(string):
    for char in ['\r\n', '\r', '\n']:
        string = string.replace(char, ' ')
    return string


class CoNaLa(BaseDataset):
    """CoNaLa dataset."""

    def _load(self, examples=100000):
        """Load a dataset into memory."""

        func_names = []
        fname = 'conala-{}.json'.format(self.dataset)
        with open(os.path.join(self.root, fname)) as f:
            data = json.load(f)

        new_data = []
        for x in data:
            snippet = x['snippet']
            tokens = tokenize_code(snippet)
            if self.adversarial_func:
                index = tokens.index('def')
                func_name = tokens[index+1]
                func_names.append(func_name)
            if self.remove_func:
                try:
                    index = tokens.index('def')
                    tokens[index + 1] = obscure(tokens[index + 1])
                except (ValueError, IndexError):
                    pass
            if x['rewritten_intent'] and (not self.use_intent):
                intent = x['rewritten_intent']
            else:
                intent = x['intent']
            snippet = ' '.join([format_str(token) for token in tokens])
            new_data.append((snippet, intent))
        data = new_data

        if self.dataset == 'train':
            fname = 'conala-mined.jsonl'
            count = 0
            with open(os.path.join(self.root, fname)) as f:
                for line in f:
                    x = json.loads(line)
                    if len(x['snippet']) > 0 and len(x['snippet']) > 0:
                        snippet = x['snippet']
                        tokens = tokenize_code(snippet)
                        if self.adversarial_func:
                            index = tokens.index('def')
                            func_name = tokens[index+1]
                            func_names.append(func_name)
                        if self.remove_func:
                            try:
                                index = tokens.index('def')
                                tokens[index + 1] = obscure(tokens[index + 1])
                            except (ValueError, IndexError):
                                pass
                        if not self.adversarial_func:
                            snippet = ' '.join([format_str(token) for token in tokens])
                        intent = x['intent']
                        data.append((snippet, intent))
                        count += 1
                    if count == examples:
                        break
        # data = [(x['snippet'], x['rewritten_intent']) for x in data]
        self.data = np.array(data, dtype=np.object)
        if self.adversarial_func:
            new_data = []
            for d in tqdm(data):
                func_name = random.sample(func_names, 1)[0]
                code = d[0]
                desc = d[1]
                index = code.index('def')
                code[index+1] = func_name
                code = ' '.join([format_str(token) for token in code])
                new_data.append((code, desc))
            data = new_data
