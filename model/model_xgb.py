#-*- coding: utf-8 -*-
 
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from generate_features import get_feat

np.random.seed(1024)

train_X = get_feat('train')
train_Y = pd.read_csv('./input/train.csv',sep='\t',header=None).values[:,-1]

test_X = get_feat('test')
test_Y = pd.read_csv('./input/test.csv',sep='\t',header=None).values[:,-1]

def evaluation(probs,label):
    preds =[]
    for x in probs:
        if x<0.25:
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


def runXGB(train_X, train_Y,test_X):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eval_metric'] = 'logloss'
    param['eta'] = 0.05
    param['max_depth'] = 6
    param['silent'] = 1
    param['min_child_weight'] = 2
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['nthread']=8
    param['seed'] = 1024 

    best_it = 5000
    nfolds = 8
    test_preds = np.zeros(test_X.shape[0])
    train_preds = np.zeros(train_X.shape[0])
    skf = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=1024)
    for cn,(ind_tr, ind_te) in enumerate(skf.split(train_X,train_Y)):
        train_x = train_X[ind_tr]
        valid_x = train_X[ind_te]
        train_y = train_Y[ind_tr]
        valid_y = train_Y[ind_te]
        
        dtrain = xgb.DMatrix(train_x, train_y)
        dvalid = xgb.DMatrix(valid_x, valid_y)

        model = xgb.train(param, dtrain,
                          num_boost_round=best_it,
                          evals=[(dtrain, 'Train'), (dvalid, 'valid')],
                          early_stopping_rounds=100,
                          verbose_eval=100
                          )

        valid_preds = model.predict(xgb.DMatrix(valid_x))
        train_preds[ind_te] = valid_preds
        test_pred = model.predict(xgb.DMatrix(test_X))
        evaluation(test_pred,test_Y)
        test_preds += test_pred
        model.save_model('xgb_fold{}.model'.format(cn))
    test_preds = test_preds/5

    print('final result and scoring...')
    evaluation(test_preds,test_Y)

    with open('./input/probs_xgb.train', 'w') as fout:
        for i,pred in enumerate(train_preds,1):
            fout.write(str(pred)+'\n')

    with open('./input/probs_xgb.test', 'w') as fout:
        for i,pred in enumerate(test_preds,1):
            fout.write(str(pred)+'\n')

if __name__ == '__main__':
    runXGB(train_X,train_Y,test_X)

