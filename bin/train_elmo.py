
import argparse

import numpy as np
import sys
import os
sys.path.append(os.curdir)

import json
from bilm.training import train, load_options_latest_checkpoint, load_vocab
from bilm.data import BidirectionalLMDataset


def main(args):
    # load the vocab
    max_character_per_word = 50
    vocab = load_vocab(args.vocab_file, max_character_per_word)

    # define the options
    batch_size = args.batch  # batch size for each GPU
    # even using only GPU still put 1 because their code is GPU based
    n_gpus = args.n_gpus

    # number of tokens in training data (this for 1B Word Benchmark)
    n_train_tokens = args.n_tokens

    options_file = os.path.join(args.options_file)

    with open(options_file, 'r') as fin:
        options = json.load(fin)

    options.update({
        'dropout': 0.1,
        'bidirectional': True,
        'all_clip_norm_val': 10.0,
        'n_epochs': args.n_epochs),  # same in paper
        'n_train_tokens': n_train_tokens,
        'batch_size': batch_size,
        'n_tokens_vocab': vocab.size,
        'unroll_steps': 20,  # same in github
        'n_negative_samples_batch': int(64*args.batch),  # 8192 / bsize 128 = 64 => 64*50 = 3200
    })

    prefix = args.train_prefix
    data = BidirectionalLMDataset(prefix, vocab, test=False,
                                  shuffle_on_load=True)

    tf_save_dir = args.save_dir
    tf_log_dir = args.save_dir
    if not os.path.exists(tf_save_dir):
        os.makedirs(tf_save_dir)
    train(options, data, n_gpus, tf_save_dir, tf_log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--train_prefix', help='Prefix for train files')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--options_file', help='Options file')
    parser.add_argument('--batch', type=int, default=50)
    parser.add_argument('--n_epochs', type=int, default=10)
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--n_tokens', type=int, default=3000000)

    args = parser.parse_args()
    main(args)
