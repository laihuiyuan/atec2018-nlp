#-*- coding: utf-8 -*-
 
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from datetime import datetime

def generate_fuzz(infile,outfile):
    start = datetime.now()
    print('generate fuzz feat,data path is',infile)
    df_data = pd.read_csv(infile,sep='\t',header=None,names=['index','sen1','sen2','label'])
    df_feat = pd.DataFrame()

    df_feat['fuzz_qratio'] = df_data.apply(lambda row: fuzz.QRatio(str(row['sen1']), str(row['sen2'])), axis=1)
    df_feat['fuzz_WRatio'] = df_data.apply(lambda row: fuzz.WRatio(str(row['sen1']), str(row['sen2'])), axis=1)
    df_feat['fuzz_partial_ratio'] = df_data.apply(lambda row: fuzz.partial_ratio(str(row['sen1']), str(row['sen2'])), axis=1)
    df_feat['fuzz_partial_token_set_ratio'] = df_data.apply(lambda row: fuzz.partial_token_set_ratio(str(row['sen1']), str(row['sen2'])), axis=1)
    df_feat['fuzz_partial_token_sort_ratio'] = df_data.apply(lambda row: fuzz.partial_token_sort_ratio(str(row['sen1']), str(row['sen2'])), axis=1)
    df_feat['fuzz_token_set_ratio'] = df_data.apply(lambda row: fuzz.token_set_ratio(str(row['sen1']), str(row['sen2'])), axis=1)
    df_feat['fuzz_token_sort_ratio'] = df_data.apply(lambda row: fuzz.token_sort_ratio(str(row['sen1']), str(row['sen2'])), axis=1)

    df_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)

#generate_fuzz('./input/train.csv','./input/train_fuzz.csv')
#generate_fuzz('./input/test.csv','./input/test_fuzz.csv')

