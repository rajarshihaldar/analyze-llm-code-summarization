"""Utilities common to all datasets."""

import abc
import os
import random

# import dgl
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class BaseDataset(Dataset, abc.ABC):
    """Base class for all datasets."""

    def __init__(self, root='data', dataset='train', only_func=False,
                 remove_func=False, remove_comments=False, use_code_tokens=False, adversarial_func=False,
                 use_intent=False, remove_keyword=False, original_tokens=False):
        """Initialize a new Dataset instance.

        Parameters:
            root (str): root directory containing the dataset
            dataset (str): one of 'train', 'valid', or 'test'
            chance (int or float): chance of returning a random description
            code_rnn_transform (callable or None): transform to apply to code
            code_gnn_transform (callable or None): transform to apply to code
            desc_rnn_transform (callable or None): transform to apply to
                code descriptions
        """
        print(f"Root = {root}")
        assert os.path.isdir(root)
        assert dataset in ('train', 'valid', 'test')
        assert not (only_func and remove_func)

        self.root = root
        self.dataset = dataset
        self.only_func = only_func
        self.remove_func = remove_func
        self.remove_comments=remove_comments
        self.use_code_tokens = use_code_tokens
        self.adversarial_func = adversarial_func
        self.use_intent = use_intent
        self.remove_keyword = remove_keyword
        self.original_tokens = original_tokens

        # Load dataset into memory
        self._load()

    def __len__(self):
        """Return the length of the dataset.

        Returns:
            int: the length of the dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """Return an index within the dataset.
        Returns a matching code and description pair.
       
        Parameters:
            idx (int): index within the dataset

        Returns:
            tuple: code, desc
        """
        code, desc = self.data[idx]
        return code, desc

    @abc.abstractmethod
    def _load(self):
        """Load a dataset into memory."""
