#-*- coding: utf-8 -*-
 
import sys
import numpy as np
import pandas as pd

def evaluation(probs,label):
    preds =[]
    for x in probs:
        if x<0.3:
            preds.append(0)
        else:
            preds.append(1)
    TP = np.count_nonzero(np.multiply(label,preds))
    TN = np.count_nonzero(np.multiply([x-1 for x in label],[x-1 for x in preds]))
    FP = np.count_nonzero(np.multiply([x-1 for x in label], preds))
    FN = np.count_nonzero(np.multiply(label,[x-1 for x in preds]))
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    accuracy = (TP + TN) / (TP + FP + TN + FN + 1e-8)
    F1_score = 2 * precision * recall / (precision + recall + 1e-8)
    print('Dev eval precision {}'.format(precision))
    print('Dev eval recall {}'.format(recall))
    print('Dev eval accuracy {}'.format(accuracy))
    print('Dev eval F1-score {}'.format(F1_score))

    return F1_score

lgb = open('./input/stack_lgb.txt').readlines()
xgb = open('./input/stack_xgb.txt').readlines()
blend_probs = [] 
for x,y in zip(lgb,xgb):
    blend_probs.append(float(x.strip())*0.5+float(y.strip())*0.5)

with open(sys.argv[1], 'w') as fout:
    for i,x in enumerate(blend_probs,1):
        if x < 0.34:
            fout.write(str(i)+'\t0\n')
        else:
            fout.write(str(i)+'\t1\n')

test_Y = pd.read_csv('./input/test.csv',sep='\t',header=None).values[:,-1]
evaluation(blend_probs,test_Y)

