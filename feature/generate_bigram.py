#-*- coding: utf-8 -*-
 
import re
import sys
import jieba
import string
import pickle
import pandas as pd
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
from generate_ngram import getBigram

path = './input/'
jieba.load_userdict('./data/UserDict.txt')
p = re.compile(r'花蕾|鲜花|花卉|呗呗|花盆|萌芽|花费|华洋|花店|花芽|花园|花荚|花篮|花印|花刷|花朵|一朵花|蓓蕾|华源|芽|华阳|华彦|华颜|这朵花|一朵花|水母|开花')

string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')

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

def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

def prepare_bigram(path,out):
    print('prepare bigram,data path is',path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('sen1,sen2\n')
        for t, row in enumerate(pd.read_csv(path, sep=',').values): 
            
            q1 = row[0].split()
            q2 = row[1].split()
            q1_bigram = getBigram(q1)
            q2_bigram = getBigram(q2)
            q1_bigram = ' '.join(q1_bigram)
            q2_bigram = ' '.join(q2_bigram)
            outfile.write('%s,%s\n' % (q1_bigram, q2_bigram))

            c+=1
        time = datetime.now()-start

    print('times:',time)

#prepare_bigram(path+'train_unigram.csv',path+'train_bigram.csv')
#prepare_bigram(path+'valid_unigram.csv',path+'valid_bigram.csv')
#prepare_bigram(path+'test_unigram.csv',path+'test_bigram.csv')
