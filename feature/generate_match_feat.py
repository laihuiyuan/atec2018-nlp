#-*- coding: utf-8 -*-
 
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime

path='./input/'

def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

df_train = pd.read_csv('./input/train_unigram.csv')
train_qs = pd.Series(df_train['sen1'].tolist() + df_train['sen2'].tolist()).astype(str)
words = (" ".join(train_qs)).lower().split()
counts = Counter(words)
weights = {word: get_weight(count) for word, count in counts.items()}

def add_word_count(df_feature, df, word):
    df_feature['q1_' + word] = df['sen1'].apply(lambda x: (word in str(x).lower())*1)
    df_feature['q2_' + word] = df['sen2'].apply(lambda x: (word in str(x).lower())*1)
    df_feature[word + '_both'] = df_feature['q1_' + word] * df_feature['q2_' + word]

def word_shares(row):

    q1_list = str(row['sen1']).lower().split()
    q1words = set(q1_list)
    if len(q1words) == 0:
        return '0:0:0:0:0:0'

    q2_list = str(row['sen2']).lower().split()
    q2words = set(q2_list)
    if len(q2words) == 0:
        return '0:0:0:0:0:0'

    words_hamming = sum(1 for i in zip(q1_list, q2_list) if i[0]==i[1])/max(len(q1_list), len(q2_list))

    q1_2gram = set([i for i in zip(q1_list, q1_list[1:])])
    q2_2gram = set([i for i in zip(q2_list, q2_list[1:])])

    shared_2gram = q1_2gram.intersection(q2_2gram)

    shared_words = q1words.intersection(q2words)
    shared_weights = [weights.get(w, 0) for w in shared_words]
    q1_weights = [weights.get(w, 0) for w in q1words]
    q2_weights = [weights.get(w, 0) for w in q2words]
    total_weights = q1_weights + q2_weights

    R1 = np.sum(shared_weights) / np.sum(total_weights) #tfidf share
    R2 = len(shared_words) / (len(q1words) + len(q2words) - len(shared_words)) #count share
    Rcosine_denominator = (np.sqrt(np.dot(q1_weights,q1_weights))*np.sqrt(np.dot(q2_weights,q2_weights)))
    Rcosine = np.dot(shared_weights, shared_weights)/Rcosine_denominator
    if len(q1_2gram) + len(q2_2gram) == 0:
        R2gram = 0
    else:
        R2gram = len(shared_2gram) / (len(q1_2gram) + len(q2_2gram))
    return '{}:{}:{}:{}:{}:{}'.format(R1, R2, len(shared_words), R2gram, Rcosine, words_hamming)

def generate_match(infile,outfile):
    start = datetime.now()
    print('generate match feat,data path is',infile)
    df_feat = pd.DataFrame()

    df_data = pd.read_csv(infile,sep=',')
    df_data['word_shares'] = df_data.apply(word_shares, axis=1)
    
    df_feat['word_match_ratio'] = df_data['word_shares'].apply(lambda x: float(x.split(':')[0]))
    df_feat['word_match_ratio_root'] = np.sqrt(df_feat['word_match_ratio'])
    df_feat['tfidf_word_match_ratio'] = df_data['word_shares'].apply(lambda x: float(x.split(':')[1]))
    df_feat['shared_count'] = df_data['word_shares'].apply(lambda x: float(x.split(':')[2]))
    df_feat['shared_2gram']     = df_data['word_shares'].apply(lambda x: float(x.split(':')[3]))
    df_feat['word_match_cosine']= df_data['word_shares'].apply(lambda x: float(x.split(':')[4]))
    df_feat['words_hamming']    = df_data['word_shares'].apply(lambda x: float(x.split(':')[5]))
    where_are_nan = np.isnan(df_feat) 
    where_are_inf = np.isinf(df_feat)
    df_feat[where_are_nan] = 0
    df_feat[where_are_inf] = 0
    
    df_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)

#generate_match(path+'train_unigram.csv',path+'train_match.csv')
#generate_match(path+'valid_unigram.csv',path+'valid_match.csv')


