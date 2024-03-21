import ast
from collections import deque

from tokenize import NUMBER, STRING, NAME, OP

import torch
import argparse, yaml, pickle
import sys, token, tokenize
from io import StringIO, BytesIO
from tqdm import tqdm

# from datasets.codesearchnet import CodeSearchNet
# from models.codebert_search import CodeBERTSearch


def tokenize_code(s):
    # Add the following imports
    # from tokenize import tokenize
    # from io import BytesIO
    result = []
    flag = True
    byte_input = BytesIO(s.encode('utf-8'))
    g = tokenize.tokenize(byte_input.readline)  # tokenize the string
    for toknum, tokval, _, _, _ in g:
        if tokval == 'utf-8':
            pass
        else:
            result.append(tokval)
    return result


def obscure(data, cipher=False):
    if cipher:
        data_altered = ""
        for letter in data:
            if letter == 'z':
                letter = 'a'
            elif letter == 'Z':
                letter = 'A'
            elif letter.isalpha():
                letter = chr(ord(letter)+1)
            data_altered += letter
    else:
        data_altered = 'f'
    return data_altered


def unobscure(data):
    data_altered = ""
    for letter in data:
        if letter == 'a':
            letter = 'z'
        elif letter == 'A':
            letter = 'Z'
        elif letter.isalpha():
            letter = chr(ord(letter)-1)
        data_altered += letter
    return data_altered


def modify_func_names(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        print("error")
        return code
    print("correct")
    for node in ast.walk(tree):
        # if isinstance(node, ast.Call):
        #     if hasattr(node.func, 'id'):
        #         node.func.id = obscure(node.func.id)
        if isinstance(node, ast.FunctionDef):
            if hasattr(node, 'name'):
                node.name = obscure(node.name)
    code = ast.unparse(tree)
    return code


def contains_hash(tok):
    if not tok:
        return False
    if tok[0] == "#":
        return True
    return False


def remove_comments_from_code(code):
    new_code = [tok for tok in code if not contains_hash(tok)]
    return new_code


def extract_comments_from_code(code):
    new_code = [tok for tok in code if contains_hash(tok)]
    return new_code


# def remove_comments_and_docstrings(source):
#     """
#     Returns 'source' minus comments and docstrings.
#     """
#     io_obj = StringIO(source)
#     out = ""
#     prev_toktype = tokenize.INDENT
#     last_lineno = -1
#     last_col = 0
#     for tok in tokenize.generate_tokens(io_obj.readline):
#         token_type = tok[0]
#         token_string = tok[1]
#         start_line, start_col = tok[2]
#         end_line, end_col = tok[3]
#         ltext = tok[4]
#         # The following two conditionals preserve indentation.
#         # This is necessary because we're not using tokenize.untokenize()
#         # (because it spits out code with copious amounts of oddly-placed
#         # whitespace).
#         if start_line > last_lineno:
#             last_col = 0
#         if start_col > last_col:
#             out += (" " * (start_col - last_col))
#         # Remove comments:
#         if token_type == tokenize.COMMENT:
#             pass
#         # This series of conditionals removes docstrings:
#         elif token_type == tokenize.STRING:
#             if prev_toktype != tokenize.INDENT:
#         # This is likely a docstring; double-check we're not inside an operator:
#                 if prev_toktype != tokenize.NEWLINE:
#                     # Note regarding NEWLINE vs NL: The tokenize module
#                     # differentiates between newlines that start a new statement
#                     # and newlines inside of operators such as parens, brackes,
#                     # and curly braces.  Newlines inside of operators are
#                     # NEWLINE and newlines that start new code are NL.
#                     # Catch whole-module docstrings:
#                     if start_col > 0:
#                         # Unlabelled indentation means we're inside an operator
#                         out += token_string
#                     # Note regarding the INDENT token: The tokenize module does
#                     # not label indentation inside of an operator (parens,
#                     # brackets, and curly braces) as actual indentation.
#                     # For example:
#                     # def foo():
#                     #     "The spaces before this docstring are tokenize.INDENT"
#                     #     test = [
#                     #         "The spaces before this string do not get a token"
#                     #     ]
#         else:
#             out += token_string
#         prev_toktype = token_type
#         last_col = end_col
#         last_lineno = end_line
#     return out


def main(cfg):
    root = cfg.get('root_csn', 'path/to/CSN/dataset')
    dataset = CodeSearchNet(root=root, dataset='train', chance=0.5)
    for batch, (code, _, desc, match) in enumerate(dataset):
        # print(code)
        transformed_code = remove_comments_and_docstrings(code)
        print(transformed_code)
        exit()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/config.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=1, type=int,
        help='seed for random number generation')
    args = parser.parse_args()
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg)