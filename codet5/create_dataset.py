import yaml, argparse, pickle
import torch
from tqdm import tqdm

from datasets.codesearchnet import CodeSearchNet
from datasets.conala import CoNaLa
from datasets.codexglue import CodeXGlue
from datasets.conala_mined import CoNaLaMined
from utils import set_seed

from transformers import OpenAIGPTTokenizer


def count_tokens(dataset):
    tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
    total_code = 0
    total_desc = 0
    for code, desc in tqdm(dataset):
        total_code += len(tokenizer.tokenize(code))
        total_desc += len(tokenizer.tokenize(desc))
    print(f"Length of Dataset = {len(dataset)}")
    print(f"Total/Average Tokens in Code = {total_code} / {total_code/len(dataset)}")
    print(f"Total/Average Tokens in Desc = {total_desc} / {total_desc/len(dataset)}")
    print(f"Total/Average Tokens = {total_code + total_desc} / {(total_code + total_desc)/len(dataset)}")

def create_csn(cfg, data_split='train'):
    dataset_name = cfg.get('dataset', 'csn')
    assert dataset_name in ['csn', 'codexglue']
    only_func = cfg.get('only_func', False)
    remove_func = cfg.get('remove_func', False)
    remove_comments = cfg.get('remove_comments', False)
    adversarial_func = cfg.get('adversarial_func', False)
    remove_keyword = cfg.get('remove_keyword', False)
    print(f"Dataset = {dataset_name}")
    if dataset_name == 'csn':
        root = cfg.get('root_csn', 'path/to/CSN/dataset')
        dataset = CodeSearchNet(root, dataset=data_split, remove_func=remove_func,
            only_func=only_func, remove_comments=remove_comments, adversarial_func=adversarial_func,
            remove_keyword=remove_keyword)
    elif dataset_name == 'codexglue':
        root = cfg.get('root_codexglue', 'path/to/CSN/dataset')
        dataset = CodeXGlue(root, dataset=data_split, remove_func=remove_func,
            only_func=only_func, remove_comments=remove_comments, adversarial_func=adversarial_func,
            remove_keyword=remove_keyword)
    # print(dataset[0])
    # print(len(dataset))
    count_tokens(dataset)
    save_path = cfg.get('dataset_pickle_path', 'datasets/saved_datasets')
    if remove_keyword:
        fname = f'{save_path}/{data_split}_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}_keyword.pkl'
    else:
        fname = f'{save_path}/{data_split}_{dataset_name}_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl'
    # pickle.dump(dataset, open(fname, 'wb'))


def create_conala(cfg, data_split='train'):
    root = cfg.get('root_conala', 'path/to/CoNaLa/dataset')
    only_func = cfg.get('only_func', False)
    remove_func = cfg.get('remove_func', False)
    remove_comments = cfg.get('remove_comments', False)
    adversarial_func = cfg.get('adversarial_func', False)
    use_intent = cfg.get('use_intent', False)
    dataset = CoNaLa(root, dataset=data_split, remove_func=remove_func,
        only_func=only_func, remove_comments=remove_comments, adversarial_func=adversarial_func, use_intent=use_intent)
    save_path = cfg.get('dataset_pickle_path', 'datasets/saved_datasets')
    print(dataset[0])
    print(len(dataset))
    if use_intent:
        pickle.dump(dataset, open(f'{save_path}/{data_split}_conala_intent_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl', 'wb'))
    else:
        pickle.dump(dataset, open(f'{save_path}/{data_split}_conala_{remove_func}_{only_func}_{remove_comments}_{adversarial_func}.pkl', 'wb'))


def main(cfg, args):
    dataset_name = cfg.get('dataset', 'csn')
    assert dataset_name in ['csn', 'conala', 'codexglue']
    if dataset_name in ['csn', 'codexglue']:
        create_csn(cfg, data_split=args.split)
    elif dataset_name == 'conala':
        create_conala(cfg, data_split=args.split)


if __name__ == '__main__':
    print("Creating Dataset")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '-c', '--config', default='configs/1.yaml',
        help='YAML file containing configuration parameters', metavar='FILE')
    parser.add_argument(
        '-s', '--seed', default=42, type=int,
        help='seed for random number generation')
    parser.add_argument(
        '--split', default='train', type=str,
        help='Dataset split: train/valid/test')
    args = parser.parse_args()
    # Set random seed for reproducibility
    set_seed(args)
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.Loader)
    main(cfg, args)