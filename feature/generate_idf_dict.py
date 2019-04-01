#-*- coding: utf-8 -*-

import pickle
import sys
import string
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle

path = './input/'


string.punctuation.__add__('!!')
string.punctuation.__add__('(')
string.punctuation.__add__(')')
string.punctuation.__add__('?')
string.punctuation.__add__('.')
string.punctuation.__add__(',')


def remove_punctuation(x):
    new_line = [ w for w in list(x) if w not in string.punctuation]
    new_line = ''.join(new_line)
    return new_line

def get_idf(df,n,smooth=1):
    idf = log((smooth + n) / (smooth + df))
    return idf

def prepare_idfdict(path,outfile,smooth=1.0,inverse=False):
    K_dict = dict()
    print('generate idf_dict,data path is',path)
    c = 0 
    start = datetime.now()
    sentences = []
    
    for t, row in enumerate(DictReader(open(path), delimiter=',')): 
        q1 = remove_punctuation(str(row['sen1'])).lower()
        q2 = remove_punctuation(str(row['sen2'])).lower()
        
        for sentence in [q1,q2]:
            for key in sentence.split(" "):
                df = K_dict.get(key,0)
                K_dict[key] = df+1
        c+=1
    n = c*2
    for key in K_dict:
        K_dict[key] = get_idf(K_dict[key] ,n,smooth=smooth)
    K_dict["default_idf"] = get_idf(0 ,n,smooth=smooth)
    end = datetime.now()
    pickle.dump(K_dict,open(outfile,'wb'),protocol=2)
    print('times:',end-start)

#prepare_idfdict('./input/train_unigram.csv','./input/idf_dict.pkl')
