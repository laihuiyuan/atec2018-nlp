#-*- coding: utf-8 -*-
import pickle
import sys
import string
import random
import numpy as np
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle

seed =1024
np.random.seed(seed)

path = "./input/"


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

def prepare_idf_dict(path,smooth=1.0):
    idf_dict = dict()

    c = 0
    for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            
        q1 = remove_punctuation(str(row['sen1']).lower())#.split(' ')
        q2 = remove_punctuation(str(row['sen2']).lower())#.split(' ')
        q1 = str(hash(q1))
        q2 = str(hash(q2))
        for key in [q1,q2]:
            df = idf_dict.get(key,0)
            df+=1
            idf_dict[key]=df
        c+=1
    n = c*2
    for key in idf_dict:
        idf_dict[key] = get_idf(idf_dict[key] ,n,smooth=smooth)
    idf_dict["default_idf"] = get_idf(0 ,n,smooth=smooth)

    return idf_dict

def get_idf(df,n,smooth=1):
    idf = log((smooth + n) / (smooth + df))        
    return idf

def generate_hashidf(infile,out):

    print('generate hash idf,data path is',infile)
    idf_dict = prepare_idf_dict(path+'train_unigram.csv')
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('question1_hash_count,question2_hash_count\n')
        for t, row in enumerate(DictReader(open(infile), delimiter=',')): 
            
            q1 = remove_punctuation(str(row['sen1']).lower())
            q2 = remove_punctuation(str(row['sen2']).lower())
            q1 = str(hash(q1))
            q2 = str(hash(q2))

            q1_idf = idf_dict.get(q1,idf_dict['default_idf'])
            q2_idf = idf_dict.get(q2,idf_dict['default_idf'])

            outfile.write('%s,%s\n' % (q1_idf, q2_idf))

            c+=1
            end = datetime.now()
    print('times:',end-start)

#generate_hashidf(path+'train_unigram.csv',path+'train_hashidf.csv')
#generate_hashidf(path+'valid_unigram.csv',path+'valid_hashidf.csv')
#generate_hashidf(path+'test_unigram.csv',path+'test_hashidf.csv')

