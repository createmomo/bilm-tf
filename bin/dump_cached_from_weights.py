import argparse
'''
ELMo usage example to write biLM embeddings for an entire dataset to
a file.
'''
import sys
import os
sys.path.append(os.curdir)

import h5py
from bilm import dump_bilm_embeddings


def main(args):
    dataset_file = args.target
    if not os.path.isfile(dataset_file):
        raise FileExistsError('invalid dataset {}'.format(dataset_file))

    # Location of pretrained LM.  Here we use the test fixtures.
    vocab_file = args.vocab
    options_file = args.options
    weight_file = args.weights

    embedding_file = args.embedding

    # Dump the embeddings to a file. Run this once for your dataset.
    dump_bilm_embeddings(
        vocab_file, dataset_file, options_file, weight_file, embedding_file
    )

    # Load the embeddings from the file -- here the 2nd sentence.
    with h5py.File(embedding_file, 'r') as fin:
        count = 0
        for i, name in enumerate(fin):
            count += 1
            if count < 5:
                second_sentence_embeddings = fin['{}'.format(i)][...]
                print (second_sentence_embeddings.shape)
        print (count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute test perplexity')
    parser.add_argument('--target', help='target')
    parser.add_argument('--vocab', 
                        default='vocab-2016-09-10.txt',
                        help='vocab file')
    parser.add_argument('--options', 
                        default='options.json',
                        help='options file')
    parser.add_argument('--weights', 
                        default='weights.hdf5',
                        help='weight file')
    parser.add_argument('--embedding',
                        help='embedding file')

    args = parser.parse_args()
    main(args)
