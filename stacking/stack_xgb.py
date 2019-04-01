#-*- coding: utf-8 -*-
 
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

np.random.seed(2017)

train_Y = pd.read_csv('./input/train.csv',sep='\t',header=None).values[:,-1]
train_cnn = np.array([x.strip() for x in open('./input/probs_cnn.train').readlines()])
train_dssm = np.array([x.strip() for x in open('./input/probs_dssm.train').readlines()])
train_bidaf = np.array([x.strip() for x in open('./input/probs_bidaf.train').readlines()])
train_deattn = np.array([x.strip() for x in open('./input/probs_deattn.train').readlines()])
train_X = np.stack([
                   train_cnn,
                   train_dssm,
                   train_bidaf,
                   train_deattn,
        ],1)

test_cnn = np.array([x.strip() for x in open('./input/probs_cnn.test').readlines()])
test_dssm = np.array([x.strip() for x in open('./input/probs_dssm.test').readlines()])
test_bidaf = np.array([x.strip() for x in open('./input/probs_bidaf.test').readlines()])
test_deattn = np.array([x.strip() for x in open('./input/probs_deattn.test').readlines()])
test_X = np.stack([
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


def runXGB(train_X, train_Y,test_X):
    n_fold = 5

    params = {}
    params["objective"] = "binary:logistic"
    params['eval_metric'] = 'logloss'
    params["eta"] = 0.02
    params["subsample"] = 0.6
    params["min_child_weight"] = 2
    params["colsample_bytree"] = 1.0
    params["max_depth"] = 6
    params["silent"] = 1
    params["nthread"] = 8
    params["seed"] = 1632

    f1 = 0.
    test_preds = np.zeros(test_X.shape[0])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
    for ind_tr, ind_te in skf.split(train_X,train_Y):
        train_x = train_X[ind_tr]
        valid_x = train_X[ind_te]
        train_y = train_Y[ind_tr]
        valid_y = train_Y[ind_te]
        
        dtrain = xgb.DMatrix(train_x, train_y)
        dvalid = xgb.DMatrix(valid_x, valid_y)

        model = xgb.train(params, dtrain,
                          num_boost_round=5000,
                          evals=[(dtrain, 'Train'), (dvalid, 'valid')],
                          early_stopping_rounds=100,
                          verbose_eval=100
                          )

        valid_preds = model.predict(xgb.DMatrix(valid_x))
        f1 += evaluation(valid_preds,valid_y)
        test_pred = model.predict(xgb.DMatrix(test_X))
        test_preds += test_pred
        model.save_model('xgb.model')
    test_preds = test_preds/5
    print('The avg f1 is',f1/5)

    #with open(sys.argv[1], 'w') as fout:
    #    for i,pred in enumerate(test_preds,1):
    #        if pred <0.3:
    #            fout.write(str(i)+'\t0\n')
    #        else:
    #            fout.write(str(i)+'\t1\n')
    
    with open('./input/stack_xgb.txt', 'w') as fout:
        for x in test_preds:
            fout.write(str(x)+'\n')

    test_Y = pd.read_csv('./input/test.csv',sep='\t',header=None).values[:,-1]
    evaluation(test_preds,test_Y)

if __name__ == '__main__':
    runXGB(train_X,train_Y,test_X)

