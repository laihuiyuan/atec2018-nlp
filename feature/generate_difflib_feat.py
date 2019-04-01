#-*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
import difflib
from datetime import datetime


def diff_ratios(row):
    seq = difflib.SequenceMatcher()
    seq.set_seqs(str(row.sen1), str(row.sen2))
    return seq.ratio()

def generate_difflib(infile,outfile):
    start = datetime.now()
    print('generate difflib,data path is',infile)
    data = pd.read_csv(infile,sep=',')
    df_feat = pd.DataFrame()
    df_feat['diff_ratios'] = data.apply(diff_ratios,axis=1)
    df_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)

#generate_difflib('./input/train_unigram.csv','./input/train_difflib.csv')
#generate_difflib('./input/valid_unigram.csv','./input/valid_difflib.csv')
#generate_difflib('./input/test_unigram.csv','./input/test_difflib.csv')

#generate_difflib('./input/train_bigram.csv','./input/train_difflib_bi.csv')
#generate_difflib('./input/valid_bigram.csv','./input/valid_difflib_bi.csv')
#generate_difflib('./input/test_unigram.csv','./input/test_difflib.csv')
