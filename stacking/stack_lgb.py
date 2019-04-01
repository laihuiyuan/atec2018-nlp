#-*- coding: utf-8 -*-
 
import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.cross_validation import StratifiedKFold

np.random.seed(1024)

train_Y = pd.read_csv('./input/train.csv',sep='\t',header=None).values[:,-1]
train_lgb = np.array([x.strip() for x in open('./input/probs_lgb.train').readlines()])
train_cnn = np.array([x.strip() for x in open('./input/probs_cnn.train').readlines()])
train_dssm = np.array([x.strip() for x in open('./input/probs_dssm.train').readlines()])
train_bidaf = np.array([x.strip() for x in open('./input/probs_bidaf.train').readlines()])
train_deattn = np.array([x.strip() for x in open('./input/probs_deattn.train').readlines()])
train_X = np.stack([
                   train_lgb,
                   train_cnn,
                   train_dssm,
                   train_bidaf,
                   train_deattn,
        ],1)

test_lgb = np.array([x.strip() for x in open('./input/probs_lgb.test').readlines()])
test_cnn = np.array([x.strip() for x in open('./input/probs_cnn.test').readlines()])
test_dssm = np.array([x.strip() for x in open('./input/probs_dssm.test').readlines()])
test_bidaf = np.array([x.strip() for x in open('./input/probs_bidaf.test').readlines()])
test_deattn = np.array([x.strip() for x in open('./input/probs_deattn.test').readlines()])
test_X = np.stack([
                  test_lgb,
                  test_cnn,
                  test_dssm,
                  test_bidaf,
                  test_deattn,
        ],1)
print(train_X.shape,test_X.shape)

def evaluation(probs,label):
    preds =[]
    for x in probs:
        if x<0.32:
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

    param = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'binary_logloss'},
        'learning_rate': 0.02,
        'feature_fraction': 1.0,
        'bagging_fraction': 0.6,
        'bagging_freq': 1,
        'verbose': 0,
        "num_leaves": 10,
        "verbose": -1,
        "min_split_gain": .1,
        "reg_alpha": .1,
        'nthread':8,
    }

    f1 = 0.
    nfolds = 8
    best_it = 5000
    test_preds = np.zeros(test_X.shape[0])
    skf = StratifiedKFold(train_Y, n_folds=nfolds, shuffle=True, random_state=1024)
    for ind_tr, ind_te in skf:
        train_x = train_X[ind_tr]
        valid_x = train_X[ind_te]
        train_y = train_Y[ind_tr]
        valid_y = train_Y[ind_te]
        
        dtrain = lgb.Dataset(train_x, train_y)
        dvalid = lgb.Dataset(valid_x, valid_y)
        model = lgb.train(param,
                      dtrain,
                      valid_sets=dvalid,
                      num_boost_round=best_it,
                      early_stopping_rounds=300,
                      verbose_eval=300)
        
        valid_preds = model.predict(valid_x,model.best_iteration)
        valid_preds = model.predict(valid_x,model.best_iteration)
        f1 += evaluation(valid_preds,valid_y)

        test_pred = model.predict(test_X, model.best_iteration)
        test_preds += test_pred
    test_preds = test_preds/nfolds
    print('The avg F1_score is',f1/nfolds)

    with open(sys.argv[1], 'w') as fout:
        for i,pred in enumerate(test_preds,1):
            if pred <0.32:
                fout.write(str(i)+'\t0\n')
            else:
                fout.write(str(i)+'\t1\n')

    #with open('./input/stack_lgb.txt', 'w') as fout:
    #    for x in test_preds:
    #        fout.write(str(x)+'\n')

    #test_Y = pd.read_csv('./input/test.csv',sep='\t',header=None).values[:,-1]
    #evaluation(test_preds,test_Y)

if __name__ == '__main__':
    runLGB(train_X,train_Y,test_X)

