#-*- coding: utf-8 -*-

import sys
import pickle 
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
from random import random,shuffle

from feat_utils import get_jaccard
from feat_utils import get_dice
from feat_utils import get_count_q1_in_q2
from feat_utils import get_ratio_q1_in_q2
from feat_utils import get_count_of_sen
from feat_utils import get_count_of_unique_sen
from feat_utils import get_ratio_of_unique_sen
from feat_utils import get_count_of_digit
from feat_utils import get_ratio_of_digit

path='./input/'

def generate_ngram_inter(path,out):
    print('generate basic features,data path is',path)
    c = 0
    start = datetime.now()
    with open(out, 'w') as outfile:
        outfile.write('jaccard,dice,count_q1_in_q2,ratio_q1_in_q2,count_of_sen1,count_of_sen2,count_of_unique_sen1,count_of_unique_sen2,ratio_of_unique_sen1,ratio_of_unique_sen2,count_of_digit_sen1,count_of_digit_sen2,ratio_of_digit_sen1,ratio_of_digit_sen2,count_of_sen_min,count_of_sen_max,count_of_unique_sen_min,count_of_unique_sen_max,ratio_of_unique_sen_min,ratio_of_unique_sen_max,count_of_digit_sen_min,count_of_digit_sen_max,ratio_of_digit_sen_min,ratio_of_digit_sen_max\n')
        for t, row in enumerate(DictReader(open(path), delimiter=',')): 
            
            sen1 = str(row['sen1']).split()
            sen2 = str(row['sen2']).split()

            jaccard = get_jaccard(sen1,sen2)
            dice = get_dice(sen1,sen2)

            count_q1_in_q2 = get_count_q1_in_q2(sen1,sen2)
            ratio_q1_in_q2 = get_ratio_q1_in_q2(sen1,sen2)

            count_of_sen1 = get_count_of_sen(sen1)
            count_of_sen2 = get_count_of_sen(sen2)

            count_of_sen_min = min(count_of_sen1,count_of_sen2)
            count_of_sen_max = max(count_of_sen1,count_of_sen2)
            
            count_of_unique_sen1 = get_count_of_unique_sen(sen1)
            count_of_unique_sen2 = get_count_of_unique_sen(sen2)
            
            count_of_unique_sen_min = min(count_of_unique_sen1,count_of_unique_sen2)
            count_of_unique_sen_max = max(count_of_unique_sen1,count_of_unique_sen2)
            
            ratio_of_unique_sen1 = get_ratio_of_unique_sen(sen1)
            ratio_of_unique_sen2 = get_ratio_of_unique_sen(sen2)
            
            ratio_of_unique_sen_min = min(ratio_of_unique_sen1,ratio_of_unique_sen2)
            ratio_of_unique_sen_max = max(ratio_of_unique_sen1,ratio_of_unique_sen2)
            
            count_of_digit_sen1 = get_count_of_digit(sen1)
            count_of_digit_sen2 = get_count_of_digit(sen2)
                        
            count_of_digit_sen_min = min(count_of_digit_sen1,count_of_digit_sen2)
            count_of_digit_sen_max = max(count_of_digit_sen1,count_of_digit_sen2)
            
            ratio_of_digit_sen1 = get_ratio_of_digit(sen1)
            ratio_of_digit_sen2 = get_ratio_of_digit(sen2)
                        
            ratio_of_digit_sen_min = min(ratio_of_digit_sen1,ratio_of_digit_sen2)
            ratio_of_digit_sen_max = max(ratio_of_digit_sen1,ratio_of_digit_sen2)
            
            
            outfile.write('%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' % (
                jaccard, dice,
                count_q1_in_q2,ratio_q1_in_q2,
                count_of_sen1,count_of_sen2,
                count_of_unique_sen1,count_of_unique_sen2,
                ratio_of_unique_sen1,ratio_of_unique_sen2,
                count_of_digit_sen1,count_of_digit_sen2,
                ratio_of_digit_sen1,ratio_of_digit_sen2,
                count_of_sen_min,count_of_sen_max,
                count_of_unique_sen_min,count_of_unique_sen_max,
                ratio_of_unique_sen_min,ratio_of_unique_sen_max,
                count_of_digit_sen_min,count_of_digit_sen_max,
                ratio_of_digit_sen_min,ratio_of_digit_sen_max,
                ))
            c+=1
        end = datetime.now()

    print('times:',end-start)

#generate_ngram_inter(path+'train_unigram.csv',path+'train_basic.csv')
#generate_ngram_inter(path+'valid_unigram.csv',path+'valid_basic.csv')
#generate_ngram_inter(path+'test_unigram.csv',path+'test_basic.csv')

#generate_ngram_inter(path+'train_bigram.csv',path+'train_basic_bi.csv')
#generate_ngram_inter(path+'valid_bigram.csv',path+'valid_basic_bi.csv')
#generate_ngram_inter(path+'test_bigram.csv',path+'test_basic_bi.csv')
