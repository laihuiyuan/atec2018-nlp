#-*- coding: utf-8 -*-
 
import pandas as pd
from datetime import datetime

path = './input/'

qid_graph = {}
def generate_qid_graph_table(row):

    hash_key1 = row["sen1"]
    hash_key2 = row["sen2"]
        
    qid_graph.setdefault(hash_key1, []).append(hash_key2)
    qid_graph.setdefault(hash_key2, []).append(hash_key1)

def pagerank(infile):

    d = 0.85
    MAX_ITER = 40
    df_train = pd.read_csv('./input/train_unigram.csv',sep=',')
    df_train.apply(generate_qid_graph_table, axis = 1)
    if infile!='./input/train.csv':
        df_data = pd.read_csv(infile,sep=',')
        df_data.apply(generate_qid_graph_table, axis = 1)

    pagerank_dict = {i:1/len(qid_graph) for i in qid_graph}
    num_nodes = len(pagerank_dict)
    
    for iter in range(0, MAX_ITER):
        for node in qid_graph:    
            local_pr = 0
            
            for neighbor in qid_graph[node]:
                local_pr += pagerank_dict[neighbor]/len(qid_graph[neighbor])
            
            pagerank_dict[node] = (1-d)/num_nodes + d*local_pr

    return pagerank_dict


def generate_pagerank(infile,outfile):
    start = datetime.now()
    print('Generate pagerank,data path is',infile)
    pagerank_dict = pagerank(infile)

    def get_pagerank_value(row):
        return pd.Series({
            "q1_pr": pagerank_dict[row["sen1"]],
            "q2_pr": pagerank_dict[row["sen2"]]
            })

    df_data = pd.read_csv(infile,sep=',')
    pagerank_feat = df_data.apply(get_pagerank_value, axis = 1)

    pagerank_feat.to_csv(outfile, index=False)
    end = datetime.now()
    print('times:',end-start)
        

#generate_pagerank(path+'train_unigram.csv',path+'train_pagerank.csv')
#generate_pagerank(path+'valid_unigram.csv',path+'valid_pagerank.csv')
