#-*- coding: utf-8 -*-
 
import gensim
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis

def generate_w2v_distinct(infile,outfile,model):
    start = datetime.now()
    print('generate distinct  w2v features,data path is',infile)

    df_feat = pd.DataFrame()
    df_data = pd.read_csv(infile,sep=',').fillna('unk')
   
    count_sen1,count_sen2,str_abs_len,avg_word_len1,avg_word_len2,diff_avg_word=[],[],[],[],[],[]
    cos, city,  canb, eucl, mink, bray,dists=[],[],[],[],[],[],[]
    for idx,row in enumerate(df_data.values):

        count1=len(row[0].split())
        count2=len(row[1].split())
        count_sen1.append(count1)
        count_sen2.append(count2)
        str_abs_len.append(abs(len(row[0])-len(row[1])))
        avg_len1=len(set(row[0].replace(' ','')))/(count1+0.001)
        avg_len2=len(set(row[1].replace(' ','')))/(count2+0.001)
        avg_word_len1.append(avg_len2)
        avg_word_len2.append(avg_len2)
        diff_avg_word.append(avg_len1 - avg_len2)

        a,b,c,d,e,f,g=[],[],[],[],[],[],[]
        a2 = [x for x in row[0].split() if model.get_id(x)!=1]
        b2 = [x for x in row[1].split() if model.get_id(x)!=1]


        if len(a2)==0 or len(b2)==0:
            a,b,c,d,e,f,g=[0],[0],[0],[0],[0],[0],[0]
        else:
            for i in range(len(a2)):
                v1 = model.embeddings[model.get_id(a2[i])]
                for j in range(len(b2)):

                    v2 = model.embeddings[model.get_id(b2[j])]
                    a.append(cosine(v1,v2))
                    b.append(cityblock(v1,v2))
                    c.append(canberra(v1,v2))
                    d.append(euclidean(v1,v2))
                    e.append(minkowski(v1,v2,3))
                    f.append(braycurtis(v1,v2))

                    v_diff = v1 - v2 
                    dist = np.sqrt(np.sum(v_diff**2))
                    g.append(dist)
                   
            if len(a)==0 or len(g)==0:
                a,b,c,d,e,f,g,h=[0],[0],[0],[0],[0],[0],[0],[0]
        #if(idx%10000==0):print(idx)
        cos.append(a)
        city.append(b)
        canb.append(c)
        eucl.append(d)
        mink.append(e)
        bray.append(f)
        dists.append(g)

    df_feat['count_sen1'] = count_sen1
    df_feat['count_sen2'] = count_sen2
    df_feat['str_abs_len'] = str_abs_len
    df_feat['avg_word_len1'] = avg_word_len1
    df_feat['avg_word_len2'] = avg_word_len2
    df_feat['diff_avg_word'] = diff_avg_word

    df_feat['w2v_cosine_max'] = [np.max(x) for x in cos]
    df_feat['w2v_cosine_min'] = [np.min(x) for x in cos]
    df_feat['w2v_cosine_mean'] = [np.mean(x) for x in cos]
    df_feat['w2v_cosine_std'] = [np.std(x) for x in cos]

    df_feat['w2v_cityblock_max'] = [np.max(x) for x in city]
    df_feat['w2v_cityblock_min'] = [np.min(x) for x in city]
    df_feat['w2v_cityblock_mean'] = [np.mean(x) for x in city]
    df_feat['w2v_cityblock_std'] = [np.std(x) for x in city]

    df_feat['w2v_canberra_max'] = [np.max(x) for x in canb]
    df_feat['w2v_canberra_min'] = [np.min(x) for x in canb]
    df_feat['w2v_canberra_mean'] = [np.mean(x) for x in canb]
    df_feat['w2v_canberra_std'] = [np.std(x) for x in canb]

    df_feat['w2v_euclidean_max'] = [np.max(x) for x in eucl]
    df_feat['w2v_euclidean_min'] = [np.min(x) for x in eucl]
    df_feat['w2v_euclidean_mean'] = [np.mean(x) for x in eucl]
    df_feat['w2v_euclidean_std'] = [np.std(x) for x in eucl]

    df_feat['w2v_minkowski_max'] = [np.max(x) for x in mink]
    df_feat['w2v_minkowski_min'] = [np.min(x) for x in mink]
    df_feat['w2v_minkowski_mean'] = [np.mean(x) for x in mink]
    df_feat['w2v_minkowski_std'] = [np.std(x) for x in mink]

    df_feat['w2v_braycurtis_max'] = [np.max(x) for x in bray]
    df_feat['w2v_braycurtis_min'] = [np.min(x) for x in bray]
    df_feat['w2v_braycurtis_mean'] = [np.mean(x) for x in bray]
    df_feat['w2v_braycurtis_std'] = [np.std(x) for x in bray]

    df_feat['w2v_dists_max'] = [np.max(x) for x in dists]
    df_feat['w2v_dists_min'] = [np.min(x) for x in dists]
    df_feat['w2v_dists_mtean'] = [np.mean(x) for x in dists]
    df_feat['w2v_dists_std'] = [np.std(x) for x in dists]

    where_are_nan = np.isnan(df_feat) 
    where_are_inf = np.isinf(df_feat)
    df_feat[where_are_nan] =  0
    df_feat[where_are_inf] = 0
    df_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)

#model = gensim.models.KeyedVectors.load_word2vec_format('cn.vector.bin', binary=True)

#with open('./vocab/vocab.data', 'rb') as fin:
#    vocab = pickle.load(fin)
#generate_w2v_distinct('./input/train_distinct.csv','./input/train_w2v_distinct.csv',vocab)
#generate_w2v_distinct('./input/test_distinct.csv','./input/test_w2v_distinct.csv',vocab)
