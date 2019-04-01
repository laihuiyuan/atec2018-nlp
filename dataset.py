# -*- coding:utf8 -*-

import os
import re
import json
import jieba
import string
import logging
import numpy as np
import pandas as pd
from feat_utils import *
from jieba import posseg
from collections import Counter, OrderedDict

from generate_features import get_feat

jieba.load_userdict('./data/UserDict.txt')

string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

p = re.compile(r'花蕾|鲜花|花卉|呗呗|花盆|萌芽|花费|华洋|花店|花芽|花园|花荚|花篮|花印|花刷|花朵|一朵花|这朵花|蓓蕾|华源|芽|华阳|华彦|华颜|水母|开花')

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

class Dataset(object):
    """
    This module implements the APIs for loading and using baidu reading comprehension dataset
    """

    def __init__(self,max_len, train_file,  test_file):
        self.logger = logging.getLogger("atec")
        self.max_len = max_len

        if train_file:
            self.train_set = self._load_dataset(train_file,True)
            self.logger.info('Train set size: {} text-pairs.'.format(len(self.train_set)))

        if test_file:
            self.test_set = self._load_dataset(test_file)
            self.logger.info('Test set size: {} text-pairs.'.format(len(self.test_set)))

    def _load_dataset(self, data_path,istrain=False):
        """
        Loads the dataset
        Args:
            data_path: the data file to load
        """
        
        data_set = []
        data = pd.read_csv(data_path,sep='\t',header=None)
        if istrain:
            features = get_feat('train')
            for line,feat in zip(data.values,features):
                sample = OrderedDict()
                tokens1 = line[1].split()
                tokens2 = line[2].split()
                sample['text1_tokens'] = tokens1 
                sample['text2_tokens'] = tokens2
                sample['base_fea'] = feat.tolist()
                sample['label'] = line[3]
                data_set.append(sample)
        else:
            features = get_feat('test')
            for line,feat in zip(data.values,features):
                sample = OrderedDict()
                line1 = list(jieba.cut(p.sub('花呗',strQ2B(line[1])),HMM=False))
                line2 = list(jieba.cut(p.sub('花呗',strQ2B(line[2])),HMM=False))
                tokens1 = [w for w in line1 if w not in string.punctuation]
                tokens2 = [w for w in line2 if w not in string.punctuation]
                sample['text1_tokens'] = tokens1
                sample['text2_tokens'] = tokens2
                sample['base_fea'] = feat.tolist()
                try:
                    sample['label'] = line[3]
                except:
                    sample['label'] = 0
                data_set.append(sample)
        return data_set
    
    def _process_data(self, data, pad_id, shuffle=False):
        data_size = len(data)
        indices = np.arange(data_size)
        if shuffle:
            np.random.shuffle(indices)
        new_data = [data[i] for i in indices]

        batch_data = {'text1_token_ids': [],
                      'text2_token_ids': [],
                      'base_features': [],
                      'text1_length':[],
                      'text2_length':[],
                      'label':[]}

        for sample in new_data:
            batch_data['label'].append(sample['label'])
            batch_data['base_features'].append(sample['base_fea'])
            batch_data['text1_token_ids'].append(sample['text1_token_ids'])
            batch_data['text2_token_ids'].append(sample['text2_token_ids'])
            batch_data['text1_length'].append(min(len(sample['text1_token_ids']), self.max_len))
            batch_data['text2_length'].append(min(len(sample['text1_token_ids']), self.max_len))
        batch_data, padded_len = self._dynamic_padding(batch_data, pad_id)
        return batch_data

    def _dynamic_padding(self, batch_data, pad_id):
        """
        Dynamically pads the batch_data with pad_id
        """
        pad_len = min(self.max_len, max(batch_data['text1_length']),max(batch_data['text2_length']))
        batch_data['text1_token_ids'] = [(ids + [pad_id] * (pad_len - len(ids)))[: pad_len]
                                           for ids in batch_data['text1_token_ids']]

        batch_data['text2_token_ids'] = [(ids + [pad_id] * (pad_len - len(ids)))[: pad_len]
                                           for ids in batch_data['text2_token_ids']]
        return batch_data, pad_len

    def word_iter(self, set_name=None):
        """
        Iterates over all the words in the dataset
        Args:
            set_name: if it is set, then the specific set will be used
        Returns:
            a generator
        """
        if set_name is None:
            data_set = self.train_set + self.dev_set + self.test_set
        elif set_name == 'train':
            data_set = self.train_set
        elif set_name == 'dev':
            data_set = self.dev_set
        elif set_name == 'test':
            data_set = self.test_set
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        if data_set is not None:
            for sample in data_set:
                for token in sample['text1_tokens']:
                    yield token
                for token in sample['text2_tokens']:
                    yield token
                
    def convert_to_ids(self, vocab):
        """
        Convert the question and passage in the original dataset to ids
        Args:
            vocab: the vocabulary on this dataset
        """
        for data_set in [self.train_set, self.test_set]:
            if data_set is None:
                continue
            for sample in data_set:
                sample['text1_token_ids'] = vocab.convert_to_ids(sample['text1_tokens'])
            for sample in data_set:
                sample['text2_token_ids'] = vocab.convert_to_ids(sample['text2_tokens'])

    def gen_mini_batches(self, set_name, batch_size, pad_id, train=False):
        """
        Generate data batches for a specific dataset (train/dev/test)
        Args:
            set_name: train/dev/test to indicate the set
            batch_size: number of samples in one batch
            pad_id: pad id
            shuffle: if set to be true, the data is shuffled.
        Returns:
            a generator for all batches
        """
        if set_name == 'train':
            data = self._process_data(self.train_set, pad_id)
        elif set_name == 'test':
            data = self._process_data(self.test_set, pad_id)
        else:
            raise NotImplementedError('No data set named as {}'.format(set_name))
        data_size = len(data['text1_length'])
        for batch_start in np.arange(0, data_size, batch_size):
            yield [
                    data['label'][batch_start: batch_start + batch_size],
                    data['text1_token_ids'][batch_start: batch_start + batch_size],
                    data['text2_token_ids'][batch_start: batch_start + batch_size],
                    data['text1_length'][batch_start: batch_start + batch_size],
                    data['text2_length'][batch_start: batch_start + batch_size],
                    data['base_features'][batch_start: batch_start + batch_size],
                ]        
