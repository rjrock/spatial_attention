'''build_vocab.py'''

import argparse
import json
import nltk
import pickle

from utils import directory

from collections import Counter
from tqdm import tqdm

from vocab import Vocabulary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--caption_path', type=str,
        default=f'{directory.annotations}/train_cap2img.json',
        help='annotation file path')
    parser.add_argument(
        '--vocab_path', type=str, default=f'{directory.vocab}/vocab.pkl',
        help='vocabulary save path')
    parser.add_argument(
        '--threshold', type=int, default=4,
        help='minimum token count threshold')
    return parser.parse_args()


def build_vocab(datafile, threshold):
    counter = Counter()

    with open(datafile, 'r') as f:
        data = json.load(f)

    for caption in tqdm(list(map(lambda x: x['caption'], data))):
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)

    tokens = [token for token, count in counter.items() if count >= threshold]
    vocab = Vocabulary()
    vocab.add_tokens(tokens)
    return vocab


def main():
    args = parse_args()
    vocab = build_vocab(datafile=args.caption_path, threshold=args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path, 'wb') as f:
        pickle.dump(vocab, f)
    print(f'Number of tokens in vocabulary: {len(vocab)}')
    print(f'Vocabulary saved to: {vocab_path}')


if __name__ == '__main__':
    main()
