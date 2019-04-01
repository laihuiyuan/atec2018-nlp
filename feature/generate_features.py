#-*- coding: utf-8 -*-

import sys
reload(sys) 
sys.setdefaultencoding('utf-8') 
import pickle
import gensim
import numpy as np
import pandas as pd
from datetime import datetime

from generate_unigram import prepare_unigram
from generate_bigram import prepare_bigram
from generate_idf_dict import prepare_idfdict
from generate_distinct import prepare_distinct

from generate_len_feat import generate_len
from generate_match_feat import generate_match
from generate_basic_feat import generate_ngram_inter
from generate_difflib_feat import generate_difflib
from generate_graph_pagerank import generate_pagerank
from generate_position_feat import generate_position 
from generate_idf_feat import generate_idfstats
from generate_hashidf_feat import generate_hashidf
from generate_fuzz_feat import generate_fuzz
from generate_tfidf_feat import generate_tfidf
from generate_w2v_feat import generate_w2v_feat 
from generate_distinct_feat import generate_w2v_distinct

path='./input/'

def generate_feat(infile,prefix='train',istrain=True):
    start = datetime.now()

    prepare_unigram(infile,path+prefix+'_unigram.csv',istrain)
    prepare_bigram(path+prefix+'_unigram.csv',path+prefix+'_bigram.csv')
    if istrain:
        generate_fuzz(path+prefix+'.raw.csv',path+prefix+'_fuzz.csv')
        prepare_idfdict(path+prefix+'_unigram.csv',path+'idf_dict.pkl')
    else:
        generate_fuzz(infile,path+prefix+'_fuzz.csv')
    prepare_distinct(path+prefix+'_unigram.csv',path+prefix+'_distinct.csv')

    generate_len(path+prefix+'_unigram.csv',path+prefix+'_len.csv')
    generate_match(path+prefix+'_unigram.csv',path+prefix+'_match.csv')
    generate_ngram_inter(path+prefix+'_unigram.csv',path+prefix+'_basic.csv')
    generate_ngram_inter(path+prefix+'_bigram.csv',path+prefix+'_basic_bi.csv')
    generate_difflib(path+prefix+'_unigram.csv',path+prefix+'_difflib.csv')
    generate_difflib(path+prefix+'_bigram.csv',path+prefix+'_difflib_bi.csv')
    generate_pagerank(path+prefix+'_unigram.csv',path+prefix+'_pagerank.csv')
    generate_position(path+prefix+'_unigram.csv',path+prefix+'_position.csv')
    generate_idfstats(path+prefix+'_unigram.csv',path+prefix+'_idfstats.csv')
    generate_hashidf(path+prefix+'_unigram.csv',path+prefix+'_hashidf.csv')
    generate_tfidf(path+prefix+'_unigram.csv',path+prefix+'_tfidf.csv')

    #with open('./vocab/vocab.data', 'rb') as fin:
    #    vocab = pickle.load(fin)
    #generate_w2v_feat(path+prefix+'_unigram.csv',path+prefix+'_w2v_feat.csv',vocab)
    #generate_w2v_distinct(path+prefix+'_distinct.csv',path+prefix+'_w2v_distinct.csv',vocab)

    end = datetime.now()
    print('times:',end-start)

def get_feat(prefix):
    
    len_feat = pd.read_csv(path+prefix+'_len.csv').values
    match_feat = pd.read_csv(path+prefix+'_match.csv').values
    basic_feat = pd.read_csv(path+prefix+'_basic.csv').values
    basic_feat_bi = pd.read_csv(path+prefix+'_basic_bi.csv').values
    difflib_feat = pd.read_csv(path+prefix+'_difflib.csv').values
    difflib_feat_bi = pd.read_csv(path+prefix+'_difflib_bi.csv').values
    pagerank_feat = pd.read_csv(path+prefix+'_pagerank.csv').values
    position_feat = pd.read_csv(path+prefix+'_position.csv').values
    idf_feat = pd.read_csv(path+prefix+'_idfstats.csv').values
    hashidf_feat = pd.read_csv(path+prefix+'_hashidf.csv').values
    fuzz_feat = pd.read_csv(path+prefix+'_fuzz.csv').values
    tfidf_feat = pd.read_csv(path+prefix+'_tfidf.csv').values
    #w2v_feat = pd.read_csv(path+prefix+'_w2v_feat.csv').values
    #distinct_feat = pd.read_csv(path+prefix+'_w2v_distinct.csv').values

    feat = np.hstack([
        len_feat,
        match_feat,
        basic_feat,
        basic_feat_bi,
        difflib_feat,
        difflib_feat_bi,
        pagerank_feat,
        position_feat,
        idf_feat,
        hashidf_feat,
        fuzz_feat,
        tfidf_feat,
        #distinct_feat,
        #w2v_feat,
        ])
    print(feat.shape)
    return feat

#generate_feat(path+'train.csv','train',True)
#get_feat('train')
#generate_feat(path+'test.csv','test',False)
#get_feat('test')
