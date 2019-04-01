#-*- coding: utf-8 -*-
 
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import StratifiedKFold

from generate_features import get_feat

np.random.seed(1024)

test_X = get_feat('test')
print(test_X.shape)

def runLGB(test_X,nfolds=8):

    test_preds = np.zeros(test_X.shape[0])
    for cn in range(nfolds):
        model = lgb.Booster(model_file='lgb_fold{}.model'.format(cn)) 
        test_pred = model.predict(test_X, model.best_iteration)
        test_preds += test_pred
    test_preds = test_preds/nfolds

    with open('./input/probs_lgb.test', 'w') as fout:
        for i,pred in enumerate(test_preds,1):
            fout.write(str(pred)+'\n')

if __name__ == '__main__':
    runLGB(test_X)

