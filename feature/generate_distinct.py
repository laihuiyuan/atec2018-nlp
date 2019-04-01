#-*- coding: utf-8 -*-
 
import sys
import pickle
import string
from csv import DictReader
from datetime import datetime
from math import exp, log, sqrt
from random import random,shuffle

path = './input/'

def distinct_terms(lst1, lst2):
    lst1 = lst1.split()
    lst2 = lst2.split()
    common = set(lst1).intersection(set(lst2))
    new_lst1 = ' '.join([w for w in lst1 if w not in common])
    new_lst2 = ' '.join([w for w in lst2 if w not in common])
    
    return (new_lst1,new_lst2)

def prepare_distinct(path,out):
    print('prepare distinct,data path is',path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('sen1,sen2\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
           
            q1 = str(row['sen1'])
            q2 = str(row['sen2'])
            coo_terms = distinct_terms(q1,q2)
            outfile.write('%s,%s\n' % coo_terms)
            c+=1
        end = datetime.now()
    print('times:',end-start)

#prepare_distinct(path+'train_unigram.csv',path+'train_distinct_u.csv')
#prepare_distinct(path+'valid_unigram.csv',path+'valid_distinct_u.csv')
