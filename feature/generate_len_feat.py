#-*- coding: utf-8 -*-
 
import numpy as np
import pandas as pd
from datetime import datetime

seed = 1024
np.random.seed(seed)

path='./input/'

def str_abs_diff_len(str1, str2):
    return abs(len(str1)-len(str2))

def str_len(str1):
    return len(str(str1))

def char_len(str1):
    str1_list = set(str(str1).replace(' ',''))
    return len(str1_list)

def word_len(str1):
    str1_list = str1.split(' ')
    return len(str1_list)

def word_match_share(row):
    q1words = {}
    q2words = {}
    for word in str(row['sen1']).lower().split():
        q1words[word] = 1
    for word in str(row['sen2']).lower().split():
        q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few sens that are nothing but stopwords
        return 0
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))*1.0/(len(q1words) + len(q2words))
    return R

def generate_len(infile,outfile):
    start = datetime.now()
    print('Generate len feat,data path is',infile)
    df_feat = pd.DataFrame()
    df_data = pd.read_csv(infile,sep=',')

    df_feat['R']=df_data.apply(word_match_share, axis=1)
    df_feat['abs_diff_len'] = df_data.apply(lambda x:str_abs_diff_len(x['sen1'],x['sen2']),axis=1)
    df_feat['common_words'] = df_data.apply(lambda x: len(set(str(x['sen1']).lower().split()).intersection(set(str(x['sen2']).lower().split()))), axis=1)

    for c in ['sen1','sen2']:
        df_feat['%s_char_len'%c] = df_data[c].apply(lambda x:char_len(x))
        
        df_feat['%s_str_len'%c] = df_data[c].apply(lambda x:str_len(x))
    
        df_feat['%s_word_len'%c] = df_data[c].apply(lambda x:word_len(x))

    df_feat['char_len_diff_ratio'] = df_feat.apply(
                lambda row: abs(row.sen1_char_len-row.sen2_char_len)/(row.sen1_char_len+row.sen2_char_len), axis=1)
    df_feat['word_len_diff_ratio'] = df_feat.apply(
                lambda row: abs(row.sen1_word_len-row.sen2_word_len)/(row.sen1_word_len+row.sen2_word_len), axis=1)

    df_feat['avg_world_len1'] = df_feat['sen1_char_len'] / df_feat['sen1_word_len']
    df_feat['avg_world_len2'] = df_feat['sen2_char_len'] / df_feat['sen2_word_len']
    df_feat['diff_avg_word'] = df_feat['avg_world_len1'] - df_feat['avg_world_len2']

    df_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)
        

#generate_len(path+'train_unigram.csv',path+'train_len.csv')
#generate_len(path+'valid_unigram.csv',path+'valid_len.csv')
#generate_len(path+'valid_unigram.csv',path+'test_len.csv')
