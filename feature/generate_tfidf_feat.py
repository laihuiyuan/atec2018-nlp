#-*- coding: utf-8 -*-
 
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def generate_tfidf(infile,outfile):
    start = datetime.now()
    print('generate word tfidf,data path is',infile)
    
    df_feat = pd.DataFrame()
    tfidf = TfidfVectorizer(ngram_range=(1, 1),min_df=3)

    df_data = pd.read_csv('./input/train_unigram.csv',sep=',')
    questions_txt = pd.Series(
            df_data['sen1'].tolist() +
            df_data['sen2'].tolist()
            ).astype(str)
    tfidf.fit(questions_txt)
    
    tfidf_sum1 = []
    tfidf_sum2 = []
    tfidf_mean1 = []
    tfidf_mean2 = []
    tfidf_len1= []
    tfidf_len2 = []

    df_data = pd.read_csv(infile,sep=',')
    for index, row in df_data.iterrows():
        tfidf_q1 = tfidf.transform([str(row.sen1)]).data
        tfidf_q2 = tfidf.transform([str(row.sen2)]).data
    
        tfidf_sum1.append(np.sum(tfidf_q1))
        tfidf_sum2.append(np.sum(tfidf_q2))
        tfidf_mean1.append(np.mean(tfidf_q1))
        tfidf_mean2.append(np.mean(tfidf_q2))
        tfidf_len1.append(len(tfidf_q1))
        tfidf_len2.append(len(tfidf_q2))

    df_feat['tfidf_sum1'] = tfidf_sum1
    df_feat['tfidf_sum2'] = tfidf_sum2
    df_feat['tfidf_mean1'] = tfidf_mean1
    df_feat['tfidf_mean2'] = tfidf_mean2
    df_feat['tfidf_len1'] = tfidf_len1
    df_feat['tfidf_len2'] = tfidf_len2 
    where_are_nan = np.isnan(df_feat) 
    where_are_inf = np.isinf(df_feat)
    df_feat[where_are_nan] = 0
    df_feat[where_are_inf] = 0
    df_feat.to_csv(outfile, index=False)
    
    end = datetime.now()
    print('times:',end-start)

#generate_tfidf('./input/train_unigram.csv','./input/train_tfidf.csv')
#generate_tfidf('./input/test_unigram.csv','./input/test_tfidf.csv')
