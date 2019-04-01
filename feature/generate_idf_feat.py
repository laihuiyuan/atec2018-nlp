#-*- coding: utf-8 -*-
 
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle
import pickle
import sys
import string

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

def mean(x):
    return sum(x)/float(len(x))

def median(x):
    len_2 = int(len(x)/2)
    return x[len_2]

def std(x):
    mean_x = mean(x)
    s = 0.0
    for xx in x:
        s+=(xx-mean_x)**2
    s/=len(x)
    s = sqrt(s)
    return s

def generate_idfstats(infile,out):
    print("Generate idf stats feat,data path is",infile)
    idf_dict = pickle.load(open(path+'idf_dict.pkl','rb'))
    K_dict = dict()
    c = 0
    start = datetime.now()
    sentences = []
    with open(out, 'w') as outfile:
        outfile.write('min_q1_idfs,max_q1_idfs,mean_q1_idfs,median_q1_idfs,std_q1_idfs,min_q2_idfs,max_q2_idfs,mean_q2_idfs,median_q2_idfs,std_q2_idfs\n')
        for t, row in enumerate(DictReader(open(infile), delimiter=',')): 
            
            q1 = remove_punctuation(str(row['sen1'])).lower()
            q2 = remove_punctuation(str(row['sen2'])).lower()
            
            q1_idfs = [idf_dict.get(key,idf_dict['default_idf']) for key in q1.split()]
            q2_idfs = [idf_dict.get(key,idf_dict['default_idf']) for key in q2.split()]

            if len(q1_idfs)==0:
                q1_idfs = [0]

            if len(q2_idfs)==0:
                q2_idfs = [0]

            min_q1_idfs = min(q1_idfs)
            max_q1_idfs = max(q1_idfs)
            mean_q1_idfs = mean(q1_idfs)
            median_q1_idfs = median(q1_idfs)
            std_q1_idfs = std(q1_idfs)

            min_q2_idfs = min(q2_idfs)
            max_q2_idfs = max(q2_idfs)
            mean_q2_idfs = mean(q2_idfs)
            median_q2_idfs = median(q2_idfs)
            std_q2_idfs = std(q2_idfs)

            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                min_q1_idfs,
                max_q1_idfs,
                mean_q1_idfs,
                median_q1_idfs,
                std_q1_idfs,
                min_q2_idfs,
                max_q2_idfs,
                mean_q2_idfs,
                median_q2_idfs,
                std_q2_idfs
                ))
            c+=1
        end = datetime.now()
        print('times:',end-start)

#create_idf_stats_features(path+'train_unigram.csv',path+'train_idfstats.csv')
#create_idf_stats_features(path+'valid_unigram.csv',path+'valid_idfstats.csv')
#create_idf_stats_features(path+'test_unigram.csv',path+'test_idfstats.csv')

