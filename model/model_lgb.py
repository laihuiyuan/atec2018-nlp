#-*- coding: utf-8 -*-
 
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import StratifiedKFold

from generate_features import get_feat

np.random.seed(1024)

train_X = get_feat('train')
train_Y = pd.read_csv('./input/train.csv',sep='\t',header=None).values[:,-1]

test_X = get_feat('test')
test_Y = pd.read_csv('./input/test.csv',sep='\t',header=None).values[:,-1]

def evalution(probs,label):
    preds = []
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


def runLGB(train_X, train_Y,test_X):
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.7,
        'bagging_freq': 2,
        'verbose': 0,
        "num_leaves": 35,
        "verbose": -1,
        "min_split_gain": .1,
        'max_depth':8,
        "reg_alpha": .1,
        'nthread':8,
    }

    best_it = 5000
    f1, nfolds = 0., 8
    test_preds = np.zeros(test_X.shape[0])
    train_preds = np.zeros(train_X.shape[0])
    skf = StratifiedKFold(train_Y, n_folds=nfolds, shuffle=True, random_state=1024)
    for cn,(ind_tr, ind_te) in enumerate(skf):
        train_x = train_X[ind_tr]
        valid_x = train_X[ind_te]
        train_y = train_Y[ind_tr]
        valid_y = train_Y[ind_te]
        
        dtrain = lgb.Dataset(train_x, train_y)
        dvalid = lgb.Dataset(valid_x, valid_y)
        model = lgb.train(params,
                      dtrain,
                      valid_sets=dvalid,
                      num_boost_round=best_it,
                      early_stopping_rounds=300,
                      verbose_eval=300)
        
        valid_preds = model.predict(valid_x,model.best_iteration)
        train_preds[ind_te] = valid_preds

        test_pred = model.predict(test_X, model.best_iteration)
        evalution(test_pred,test_Y)
        test_preds += test_pred
        model.save_model('lgb_fold{}.model'.format(cn))

    print('The avg F1_score is',f1/nfolds)
    test_preds = test_preds/nfolds
    evalution(test_preds,test_Y)

    with open('./input/probs_lgb.train', 'w') as fout:
        for i,pred in enumerate(train_preds,1):
            fout.write(str(pred)+'\n')

    with open('./input/probs_lgb.test', 'w') as fout:
        for i,pred in enumerate(test_preds,1):
            fout.write(str(pred)+'\n')

if __name__ == '__main__':
    runLGB(train_X,train_Y,test_X)

