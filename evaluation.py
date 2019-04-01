#-*- coding: utf-8 -*-

import sys
import numpy as np 
import pandas as pd


label = pd.read_csv(sys.argv[1],sep='\t',header=None).values[:,-1]
preds = []
with open(sys.argv[2]) as fout:
    for line in fout:
        if float(line.strip())<0.3:
            preds.append(0)
        else:
            preds.append(1)
        #items=line.strip().split()
        #preds.append(int(items[0]))

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
