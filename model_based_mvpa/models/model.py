#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho
@contact: cjfwndnsl@gmail.com
@last modification: 2020.11.02
"""

import numpy as np
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.utils import Sequence
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import  ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter
from scipy.stats import ttest_1samp

class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(X.shape[0])
        self.on_epoch_end()
        
    # for printing the statistics of the function
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        "Denotes the number of batches per epoch"
        
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):  # index : batch no.
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        niis = [self.X[i] for i  in indexes]
        targets = [self.y[i] for i in indexes]
        niis = np.array(niis)
        targets = np.array(targets)

        return niis, targets  # return batch


def mvpa_regression_MLP(
                        # data
                        X, y,
                        # model configuration
                        layer_dims = [1024,1024],
                        act_fun = 'linear',
                        dropout_rate = 0.5,
                        # fitting configuration
                        epochs = 100,
                        patience = 10,
                        batch_size = 64,
                        # other configuration
                        repeat_N = 15,
                        verbose = 0):
    
    X = X.reshape(-1, X.shape[-1])
    y = y.flatten()
    
    if verbose > 0:
        print(f'INFO [MLP]: start running ')
        
    coeffs = []

    for repeat_i in range(repeat_N):
        ids = np.arange(X.shape[0])
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=repeat_i)
        train_steps = len(train_ids) // batch_size
        valid_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        y_train = y[train_ids]
        X_test = X[test_ids]
        y_test = y[test_ids]

        train_generator = DataGenerator(X_train, y_train, batch_size, shuffle=True)
        valid_generator = DataGenerator(X_test, y_test, batch_size, shuffle=False)
        
        bst_model_path = f'temp_{repeat_i}_best_mlp.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True,monitor='val_loss',mode='min',)

        #adam= optimizers.Adam(lr=0.0005, decay=0)
        
        model = Sequential()
        model.add(Dense(layer_dims[0], activation=act_fun, input_shape=(X.shape[-1],),use_bias=False)) 
        model.add(Dropout(dropout_rate))
        
        for dim in layer_dims[1:]:

            model.add(Dense(dim, activation=act_fun, use_bias=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='linear',use_bias=True))
        model.compile(loss='mse', optimizer='adam',)

        model.fit_generator(generator=train_generator, validation_data=valid_generator,
                            steps_per_epoch=train_steps, validation_steps=valid_steps,
                            epochs=epochs,
                            verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=patience), model_checkpoint])
        model.load_weights(bst_model_path)

        pred = model(X_test).numpy()
        mse = ((pred-y_test)**2).mean()
        if verbose > 0:
            print(f'INFO [{repeat_i+1}/{repeat_N}] - mse: {mse:.03f}')
            
        weights = []
        for layer in model.layers:
            if 'dense' not in layer.name:
                continue
            weights.append(layer.get_weights()[0])

        coeff = weights[0]
        for weight in weights[1:]:

            coeff = np.matmul(coeff,weight)
        coeffs.append(coeff.ravel())

    coeffs = np.array(coeffs)
    
    # coeffs : repeat_N x voxel #
    
    return coeffs

def mvpa_regression_penalized_linear(
                        # data
                        X, y,
                        # model configuration
                        alpha = 0.001, # mixing parameter
                        lambda_par = 0.8, # shrinkage parameter
                        # fitting configuration
                        epochs = 100,
                        patience = 30,
                        batch_size = 256,
                        # other configuration
                        repeat_N = 15,
                        verbose = 0):
    
    X = X.reshape(-1, X.shape[-1])
    y = y.flatten()
    
    if verbose > 0:
        print(f'INFO [MLP]: start running ')
        
    coeffs = []

    for repeat_i in range(repeat_N):
        ids = np.arange(X.shape[0])
        train_ids, test_ids = train_test_split(ids, test_size=0.2, random_state=repeat_i)
        train_steps = len(train_ids) // batch_size
        valid_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        y_train = y[train_ids]
        X_test = X[test_ids]
        y_test = y[test_ids]

        train_generator = DataGenerator(X_train, y_train, batch_size, shuffle=True)
        valid_generator = DataGenerator(X_test, y_test, batch_size, shuffle=False)

        bst_model_path = f'temp_{repeat_i}_best_mlp.h5'
        model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True,monitor='val_loss',mode='min',)

        #adam= optimizers.Adam(lr=0.0005, decay=0)
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=(X.shape[-1],),
                        use_bias=True,
                        kernel_regularizer=l1_l2(lambda_par*alpha,lambda_par*(1-alpha)/2),)) 
        model.compile(loss='mse', optimizer='adam',)

        model.fit_generator(generator=train_generator, validation_data=valid_generator,
                            steps_per_epoch=train_steps, validation_steps=valid_steps,
                            epochs=epochs,
                            verbose=0, callbacks=[EarlyStopping(monitor='val_loss', patience=patience), model_checkpoint])
        model.load_weights(bst_model_path)
    
        pred = model(X_test).numpy()
        mse = ((pred-y_test)**2).mean()
        if verbose > 0:
            print(f'INFO [{repeat_i+1}/{repeat_N}] - mse: {mse:.03f}')
        coeff = model.layers[0].get_weights()[0] 
        coeffs.append(coeff)

    coeffs = np.array(coeffs)
    
    # coeffs : repeat_N x voxel #
    
    return coeffs

