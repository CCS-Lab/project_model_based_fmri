#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho, Yedarm Seong
@contact: cjfwndnsl@gmail.com
          mybirth0407@gmail.com
@last modification: 2020.11.16
"""

import numpy as np
from tqdm import tqdm
import os

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import  ModelCheckpoint,EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from scipy.stats import ttest_1samp
from ..data import loader
from glmnet import ElasticNet

import logging


logging.basicConfig(level=logging.INFO)
 
        
def mlp_regression(X, y, # input data
                   # specification for model design & training
                   temp_path=None,
                   layer_dims=[1024,1024],
                   activation_func='linear',
                   dropout_rate=0.5,
                   epochs=100,
                   patience=10,
                   batch_size=64,
                   val_ratio=0.2,
                   N=15,
                   verbose=0,
                   max_use_sample_N=30000):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        ids = np.arange(X.shape[0])
        if X.shape[0] > max_use_sample_N:
            np.random.shuffle(ids)
            ids = ids[:max_use_sample_N]
        train_ids, test_ids = train_test_split(
            ids, test_size=val_ratio, random_state=42 + (i * i)
        )
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)
        
        if temp_path is None:
            temp_path = os.getcwd()
        bst_model_path = os.path.join(temp_path, f'temp_id{random.random():.03f}_best_mlp.h5')
        
        mc = ModelCheckpoint(
            bst_model_path,
            save_best_only=True, save_weights_only=True,
            monitor='val_loss', mode='min'
        )
        es = EarlyStopping(monitor='val_loss', patience=patience)
        #adam= optimizers.Adam(lr=0.0005, decay=0)
        
        model = Sequential()
        model.add(Dense(layer_dims[0],
                        activation=activation_func, input_shape=(X.shape[-1],),
                        use_bias=False)
        )
        model.add(Dropout(dropout_rate))
        
        for dim in layer_dims[1:]:
            model.add(Dense(dim, activation=activation_func, use_bias=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='linear', use_bias=True))
        model.compile(loss='mse', optimizer='adam')

        model.fit(train_generator, batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  callbacks=[mc, es],
                  validation_data=val_generator,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps)

        model.load_weights(bst_model_path)

        # results = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        error = mean_squared_error(y_pred, y_test)

        if verbose > 0:
            logging.info(f'[{i+1}/{N}] - mse: {error:.04f}')
            
        weights = []
        for layer in model.layers:
            if 'dense' not in layer.name:
                continue
            weights.append(layer.get_weights()[0])

        coef = weights[0]
        for weight in weights[1:]:
            coef = np.matmul(coef,weight)
        coefs.append(coef.ravel())
        os.remove(bst_model_path)

    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs
    

def penalized_linear_regression(X, y,
                                temp_path=None,
                                alpha=0.001, # mixing parameter
                                lambda_par=0.8, # shrinkage parameter
                                epochs=100,
                                patience=30,
                                batch_size=64,
                                val_ratio=0.2,
                                N=15,
                                verbose=0,
                                max_use_sample_N=30000):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        ids = np.arange(X.shape[0])
        if X.shape[0] > max_use_sample_N:
            np.random.shuffle(ids)
            ids = ids[:max_use_sample_N]
        
        train_ids, test_ids = train_test_split(
            ids, test_size=val_ratio, random_state=42 + (i * i))
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)

        if temp_path is None:
            temp_path = os.getcwd()
        bst_model_path = os.path.join(temp_path, f'temp_id{random.random():.03f}_best_pnl.h5')
        
        es = EarlyStopping(monitor='val_loss', patience=patience)
        mc = ModelCheckpoint(bst_model_path,
                             save_best_only=True,
                             save_weights_only=True,
                             monitor='val_loss',
                             mode='min')

        kernel_regularizer = l1_l2(lambda_par * alpha, lambda_par * (1 - alpha) / 2)
        #adam= optimizers.Adam(lr=0.0005, decay= 0)
        model = Sequential()
        model.add(Dense(1, activation='linear', input_shape=(X.shape[-1]),
                        use_bias=True, kernel_regularizer=kernel_regularizer))
        model.compile(loss='mse', optimizer='adam')

        model.fit(train_generator,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  callbacks=[mc, es],
                  validation_data=val_generator,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps)

        model.load_weights(bst_model_path)

        y_pred = model.predict(X_test)
        error = mean_squared_error(y_pred, y_test)

        if verbose > 0:
            logging.info(f'[{i+1}/{N}] - mse: {error:.04f}')

        coef = model.layers[0].get_weights()[0] 
        coefs.append(coef.ravel())
        os.remove(bst_model_path)
        
    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs


def elasticnet(X, y,
               alpha=0.001,
               n_jobs=16,
               N=3,
               verbose=0,
               max_use_sample_N=30000):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        ids = np.arange(X.shape[0])
        if X.shape[0] > max_use_sample_N:
            np.random.shuffle(ids)
            ids = ids[:max_use_sample_N]
        X_data = X #[ids]
        y_data  = y #[ids]
        model = ElasticNet(alpha = alpha, n_jobs =n_jobs)
        model = model.fit(X_data,y_data)
        y_pred = model.predict(X_data).flatten()
        error = mean_squared_error(y_pred, y_data)
        
        coefs.append(model.coef_.ravel())
        
        if verbose > 0:
            logging.info(f'[{i+1}/{N}] - lambda_best: {model.lambda_best_:.03f}/ mse: {error:.04f}')
    
    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs

