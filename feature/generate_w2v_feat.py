#-*- coding: utf-8 -*-
 
import gensim
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, jaccard,cityblock, canberra, euclidean, minkowski, braycurtis

def sen2sim(row,similarity,model):
    sims = []
    vector1 = [model[w] for w in row['sen1'].split() if w in model.vocab]
    vector2 = [model[w] for w in row['sen2'].split() if w in model.vocab]
    for v1 in vector1:
        for v2 in vector2:
            sims.append(similarity(v1,v2))
    if len(sims)==0:
        sims = [0.0]
    return sims

def sent2vec(sen,model):
    words = sen.split()
    M = []
    for w in words:
        try:
            M.append(model[model.get_id(w)])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / (np.sqrt((v ** 2).sum())+1e-6)

def generate_w2v_feat(infile,outfile,model):
    start = datetime.now()
    print('generate w2v features,data path is',infile)

    df_feat = pd.DataFrame()
    df_data = pd.read_csv(infile,sep=',').fillna('unk')

    cos, city, jacc, canb, eucl, mink, bray,dists=[],[],[],[],[],[],[],[]
    for idx,row in enumerate(df_data.values):

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
                a,b,c,d,e,f,g=[0],[0],[0],[0],[0],[0],[0]
        #if(idx%10000==0):print(idx)
        cos.append(a)
        city.append(b)
        canb.append(c)
        eucl.append(d)
        mink.append(e)
        bray.append(f)
        dists.append(g)

    sen1_vectors = np.zeros((df_data.shape[0], 300))
    for i, q in enumerate(df_data.sen1.values):
        sen1_vectors[i, :] = sent2vec(q,model)
        
    sen2_vectors  = np.zeros((df_data.shape[0], 300))
    for i, q in enumerate(df_data.sen2.values):
        sen2_vectors[i, :] = sent2vec(q,model)

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

    df_feat['w2v_cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(sen1_vectors), np.nan_to_num(sen2_vectors))]
    df_feat['w2v_skew_q1vec'] = [skew(x) for x in np.nan_to_num(sen1_vectors)]
    df_feat['w2v_skew_q2vec'] = [skew(x) for x in np.nan_to_num(sen2_vectors)]
    df_feat['w2v_kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(sen1_vectors)]
    df_feat['w2v_kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(sen2_vectors)]
    where_are_nan = np.isnan(df_feat) 
    where_are_inf = np.isinf(df_feat)
    df_feat[where_are_nan] = 0
    df_feat[where_are_inf] = 0
    df_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)

#model = gensim.models.KeyedVectors.load_word2vec_format('cn.vector.bin', binary=True)

#with open('./vocab/vocab.data', 'rb') as fin:
#    vocab = pickle.load(fin)
#generate_w2v_feat('./input/train_unigram.csv','./input/train_w2v_feat.csv',vocab)
#generate_w2v_feat('./input/test_unigram.csv','./input/test_w2v_feat.csv',vocab)
