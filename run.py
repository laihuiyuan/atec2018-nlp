# -*- coding:utf8 -*-

import sys

sys.path.append('..')

import os
import pickle
import argparse
import logging
from dataset import Dataset
from vocab import Vocab

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser('aetc dataset')
    parser.add_argument('--prepare', action='store_true',
                        help='create the directories, prepare the vocabulary and embeddings')
    parser.add_argument('--train', action='store_true',
                        help='train the model')
    parser.add_argument('--predict', action='store_true',
                        help='predict the answers for test set with trained model')
    parser.add_argument('--gpu', type=str, default='0',
                        help='specify gpu device')

    path_settings = parser.add_argument_group('some settings')
    path_settings.add_argument('--max_len', type=int, default=20,
                                help='max length of passage')
    path_settings.add_argument('--pre_embed',
                               default='/home/lawe/myworks/w2v/model/cn.vector2.bin',
                               help='the pretrained word_embedding data')
    path_settings.add_argument('--train_files',
                               default='./input/train.csv',
                               help='the preprocessed train data')
    path_settings.add_argument('--test_files',
                               default='./data/test.csv',
                               help='the test data')
    path_settings.add_argument('--vocab_dir', default='./vocab/',
                               help='the dir to save vocabulary')
    path_settings.add_argument('--log_path',
                               help='path of the log file. If not set, logs are printed to console')
    return parser.parse_args()

def prepare(args):
    """
    checks data, creates the directories, prepare the vocabulary and embeddings
    """
    logger = logging.getLogger("atec")
    logger.info('Checking the data files...')
    for data_path in [args.train_files,args.test_files]:
        assert os.path.exists(data_path), '{} file does not exist.'.format(data_path)

    logger.info('Building vocabulary...')
    if not os.path.exists(args.vocab_dir):
        os.makedirs(dir_path)
    brc_data = Dataset(args.max_len,args.train_files,  args.test_files)
    vocab = Vocab(lower=True)
    for word in brc_data.word_iter('train'):
        vocab.add(word)

    unfiltered_vocab_size = vocab.size()
    vocab.filter_tokens_by_cnt(min_cnt=1)
    filtered_num = unfiltered_vocab_size - vocab.size()
    logger.info('After filter {} tokens, the final vocab size is {}'.format(filtered_num,
                                                                            vocab.size()))
    logger.info('Assigning embeddings...')
    num=vocab.load_pretrained_embeddings(args.pre_embed)
    logger.info('Null word embeddings {}'.format(num))

    logger.info('Saving vocab...')
    with open(os.path.join(args.vocab_dir, 'vocab.data'), 'wb') as fout:
        pickle.dump(vocab, fout,protocol=2)

    logger.info('Done with preparing!')

def run():
    """
    Prepares and runs the whole system.
    """
    args = parse_args()

    logger = logging.getLogger("atec")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if args.log_path:
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.info('Running with args : {}'.format(args))

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    if args.prepare:
        prepare(args)
    if args.train:
        train(args)
    if args.predict:
        predict(args)

if __name__ == '__main__':
    run()
