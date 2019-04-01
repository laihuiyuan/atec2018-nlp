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

from keras import backend as K
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.engine.topology import Layer
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import concatenate
from keras import initializers, regularizers, constraints
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input,Embedding, CuDNNLSTM, Concatenate, Multiply,Dense
from keras.layers import Dropout, Lambda, Maximum, Subtract,BatchNormalization

from dataset import Dataset

np.random.seed(1024)

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + 0.000001, K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        # print weigthted_input.shape
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

def evaluation(probs,label):
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

def build_model(emb_matrix, maxlen):
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=maxlen,
        trainable=False
    )

    lstm0 = CuDNNLSTM(300, return_sequences=True)
    lstm1 = Bidirectional(CuDNNLSTM(150, return_sequences=True))
    lstm2 = CuDNNLSTM(300)
    attn1 = Attention(maxlen)
    attn2 = Attention(maxlen)

    seq1 = Input(shape=(maxlen,))
    seq2 = Input(shape=(maxlen,))

    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    lstm1a = lstm1(emb1)
    lstm1b = lstm1(emb2)

    lstm2a = lstm2(lstm0(lstm1a))
    lstm2b = lstm2(lstm0(lstm1b))

    v1 = Concatenate()([attn1(lstm1a),lstm2a])
    v2 = Concatenate()([attn2(lstm1b),lstm2b])

    feat_input = Input(shape=(data_feat.shape[1],))
    feat_dense = BatchNormalization()(feat_input)
    feat_dense = Dense(150, activation='relu')(feat_dense)
   
    mul = Multiply()([v1, v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1, v2]))
    maximum = Maximum()([Multiply()([v1, v1]), Multiply()([v2, v2])])
    #sub2 = Lambda(lambda x: K.abs(x))(Subtract()([lstm2a, lstm2b]))

    merge = Concatenate()([mul, sub, maximum,feat_dense])
    merge = Dropout(0.2)(merge)
    merge = BatchNormalization()(merge)
    merge = Dense(300, activation='relu')(merge)

    merge = Dropout(0.2)(merge)
    merge = BatchNormalization()(merge)
    res = Dense(1, activation='sigmoid')(merge)

    model = Model(inputs=[seq1,seq2,feat_input], outputs=res)
    model.compile(optimizer=Adam(lr=0.001), loss="binary_crossentropy",metrics=['acc'])

    return model

MAX_LEN = 20
with open(os.path.join('./vocab/vocab.data'), 'rb') as fin:
    vocab = pickle.load(fin)
brc_data = Dataset(MAX_LEN, './input/train.csv','./input/test.csv')
brc_data.convert_to_ids(vocab)
train_data = brc_data._process_data(brc_data.train_set,vocab.get_id(vocab.pad_token))
test_data = brc_data._process_data(brc_data.test_set,vocab.get_id(vocab.pad_token))

data_1 = np.array(train_data['text1_token_ids'])
data_2 = np.array(train_data['text2_token_ids'])
data_feat = np.array(train_data['base_features'])
labels = np.array(train_data['label'])

test_data_1 = np.array(test_data['text1_token_ids'])
test_data_2 = np.array(test_data['text2_token_ids'])
test_data_feat = np.array(test_data['base_features'])
test_Y = np.array(test_data['label'])

print(data_feat.shape,test_data_feat.shape)

embedding_matrix = vocab.embeddings 

nfolds = 8
re_weight = False
test_preds = np.zeros(test_data_1.shape[0])
train_preds = np.zeros(data_1.shape[0])
folds = KFold(data_1.shape[0], n_folds = nfolds, shuffle = True, random_state = 1024)

for curr_fold, (idx_train, idx_val) in enumerate(folds):

    data_1_train = data_1[idx_train]
    data_2_train = data_2[idx_train]
    data_f_train = data_feat[idx_train]
    labels_train = labels[idx_train]

    data_1_val = data_1[idx_val]
    data_2_val = data_2[idx_val]
    data_f_val = data_feat[idx_val]
    labels_val = labels[idx_val]

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.55 
        weight_val[labels_val==0] = 0.45

    if re_weight:
        class_weight = {0: 0.45, 1: 0.55}
    else:
        class_weight = None
    
    model = build_model(embedding_matrix,data_1.shape[1],)
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    bst_model_path = 'dssm_fold{}.h5'.format(curr_fold)
    model_checkpoint = ModelCheckpoint(bst_model_path,
                                       save_best_only=True,
                                       save_weights_only=True)
    
    print('\n',bst_model_path, "curr_fold:", curr_fold)
    
    hist = model.fit([data_1_train, data_2_train,data_f_train],
                     labels_train, 
                     validation_data=([data_1_val, data_2_val,data_f_val], labels_val, weight_val),
                     epochs=200,
                     batch_size=256,
                     shuffle=True, 
                     class_weight=class_weight,
                     callbacks=[early_stopping, model_checkpoint],
                     verbose = 2)

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    
    val_pred = model.predict([data_1_val, data_2_val, data_f_val], batch_size=2048, verbose=2).ravel()
    train_preds[idx_val] = val_pred
    
    preds = model.predict([test_data_1, test_data_2,test_data_feat], batch_size=2048, verbose=2).ravel()
    evaluation(preds,test_Y)
    test_preds += preds

print('final result and score...')
test_preds = test_preds/nfolds
evaluation(test_preds,test_Y)

with open('./input/probs_dssm.train', 'w') as fout:
    for i,pred in enumerate(train_preds,1):
        fout.write(str(pred)+'\n')

with open('./input/probs_dssm.test', 'w') as fout:
    for i,pred in enumerate(test_preds,1):
        fout.write(str(pred)+'\n')

