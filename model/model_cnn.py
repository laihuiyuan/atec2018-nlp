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
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints

from dataset import Dataset

np.random.seed(1)

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

        eij = K.relu(eij)

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

def build_model(emb_matrix, max_sequence_length):
    
    # The embedding layer containing the word vectors
    emb_layer = Embedding(
        input_dim=emb_matrix.shape[0],
        output_dim=emb_matrix.shape[1],
        weights=[emb_matrix],
        input_length=max_sequence_length,
        trainable=False
    )
    
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2, padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4, padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6, padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(max_sequence_length,))
    seq2 = Input(shape=(max_sequence_length,))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

    # self attention
    attn1 = Attention(20)
    attn2 = Attention(20)
    attn3 = Attention(20)
    attn4 = Attention(20)
    attna = attn1(emb1)
    attnb = attn1(emb2)

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    attn1a = attn2(conv1a)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    attn1b = attn2(conv1b)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, attn1a, attna])
    mergeb = concatenate([glob1b, glob2b, glob3b, attn1b, attnb])
  
    diff = Lambda(lambda x: K.abs(x[0] - x[1]), output_shape=(4 * 128 + 300,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1], output_shape=(4 * 128 + 300,))([mergea, mergeb])

    feat_input = Input(shape=(data_feat.shape[1],))
    feat_dense = BatchNormalization()(feat_input)
    feat_dense = Dense(150, activation='relu')(feat_dense)

    # Merge the Magic and distance features with the difference layer
    merge = concatenate([diff, mul, feat_dense])
    x = Dropout(0.5)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    #x = Dropout(0.3)(x)
    #x = BatchNormalization()(x)
    #x = Dense(300, activation='relu')(x)

    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[seq1, seq2, feat_input], outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    #model.summary()

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
    bst_model_path = 'cnn_fold{}.h5'.format(curr_fold)
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

with open('./input/probs_cnn.train', 'w') as fout:
    for i,pred in enumerate(train_preds,1):
        fout.write(str(pred)+'\n')

with open('./input/probs_cnn.test', 'w') as fout:
    for i,pred in enumerate(test_preds,1):
        fout.write(str(pred)+'\n')
