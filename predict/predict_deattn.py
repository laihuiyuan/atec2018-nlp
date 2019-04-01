#-*- coding: utf-8 -*-
 
import os
import re
import sys
import csv
import codecs
import pickle
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold

import keras.backend as K
from keras.layers import *
from keras.activations import softmax
from keras.models import Model
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ModelCheckpoint

from dataset import Dataset

np.random.seed(1)

def evaluation(probs,label):
    preds = []
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

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape

def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_

def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_

def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_

def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_

def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def build_model(emb_matrix,maxlen,projection_dim=300,
                projection_hidden=0, projection_dropout=0.2,
                compare_dim=500, compare_dropout=0.2,
                dense_dim=300, dense_dropout=0.2,
                lr=1e-3, activation='elu'):
    # Based on: https://arxiv.org/abs/1606.01933

    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=maxlen,
        trainable=False)

    seq1 = Input(name='seq1', shape=(maxlen,))
    seq2 = Input(name='seq2', shape=(maxlen,))

    seq1_embed = emb_layer(seq1)
    seq2_embed = emb_layer(seq2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    seq1_encoded = time_distributed(seq1_embed, projection_layers)
    seq2_encoded = time_distributed(seq2_embed, projection_layers)

    # Attention
    seq1_aligned, seq2_aligned = soft_attention_alignment(seq1_encoded, seq2_encoded)

    # Compare
    seq1_combined = Concatenate()([seq1_encoded, seq2_aligned, submult(seq1_encoded, seq2_aligned)])
    seq2_combined = Concatenate()([seq2_encoded, seq1_aligned, submult(seq2_encoded, seq1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    seq1_compare = time_distributed(seq1_combined, compare_layers)
    seq2_compare = time_distributed(seq2_combined, compare_layers)

    # Aggregate
    seq1_rep = apply_multiple(seq1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    seq2_rep = apply_multiple(seq2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(1000,))([seq1_rep, seq2_rep])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(1000,))([seq1_rep, seq2_rep])

    # some feature
    feat_input = Input(shape=(data_feat.shape[1],))
    feat_dense = BatchNormalization()(feat_input)
    feat_dense = Dense(150, activation='relu')(feat_dense)

    # Classifier
    merged = Concatenate()([diff, mul, feat_dense])
    #merged = Concatenate()([seq1_rep, seq2_rep,feat_input])

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[seq1, seq2, feat_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',metrics=['accuracy'])
    #model.summary()
    return model

MAX_LEN = 20
with open(os.path.join('./vocab/vocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)
brc_data = Dataset(MAX_LEN, './input/train.csv',  sys.argv[1])
brc_data.convert_to_ids(vocab)
train_data = brc_data._process_data(brc_data.train_set,vocab.get_id(vocab.pad_token),True)
test_data = brc_data._process_data(brc_data.test_set,vocab.get_id(vocab.pad_token),False)

data_1 = np.array(train_data['text1_token_ids'])
data_2 = np.array(train_data['text2_token_ids'])
data_feat = np.array(train_data['base_features'])
labels = np.array(train_data['label'])

test_data_1 = np.array(test_data['text1_token_ids'])
test_data_2 = np.array(test_data['text2_token_ids'])
test_data_feat = np.array(test_data['base_features'])
test_Y = np.array(test_data['label'])

embedding_matrix = vocab.embeddings 

nfolds = 8
test_preds = np.zeros(test_data_1.shape[0])
folds = KFold(data_1.shape[0], n_folds = nfolds, shuffle = True, random_state = 1024)
for curr_fold, (idx_train, idx_val) in enumerate(folds):
    model = build_model(embedding_matrix,data_1.shape[1])
    model.load_weights('deattn_fold{}.h5'.format(curr_fold))
    pred = model.predict([test_data_1, test_data_2,test_data_feat], batch_size=2048, verbose=1).ravel()
    test_preds += pred
test_preds = test_preds/nfolds

with open('./input/probs_deattn.test', 'w') as fout:
    for i,pred in enumerate(test_preds,1):
        fout.write(str(pred)+'\n')
evaluation(test_preds,test_Y)
