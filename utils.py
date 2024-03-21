import random
import numpy as np
import torch


OPERATORS = ['>', '<', '<=', '>=', '+', '-', '*', '/', '%', '**', '//', '=', '+=', '-=', '==', '*=', '/=', '%=', '//=', '!=', '&=', '|=', '^=', '>>=', '<<=']
PUNCTUATIONS = ['{', '}', '(', ')', '[', ']', ',', '.', ';', ':']


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
