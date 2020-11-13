#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho, Yedarm Seong
@contact: cjfwndnsl@gmail.com
          mybirth0407@gmail.com
@last modification: 2020.11.03
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


def get_map(train_generator,
            val_generator,
            model,
            model2actmap_fun,
            val_threshold):
    
    train_generator.on_epoch_end()
    for i in range(len(train_generator)):
        X,y = train_generator.__getitem__(i)
        model.fit(X,y,verbose=0)
        
    preds = []
    answers = []
    for i in range(len(val_generator)):
        X,y = val_generator.__getitem__(i)
        preds.append(model.predict(X))
        answers.append(y)
        
    preds = np.concatenate(preds,0).flatten()
    answers = np.concatenate(answers,0).flatten()
    
    mse = mean_squared_error(preds, answers)
    if mse < val_threshold :
        actmap = model2actmap_fun(model)
    else:
        actmap = None
        
    return mse, actmap


def mlp_regression_v2(X, y,
                   save_path=None,
                   layer_dims=[1024,1024],
                   activation_func='linear',
                   dropout_rate=0.5,
                   epochs=100,
                   patience=10,
                   batch_size=64,
                   N=15,
                   verbose=0):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        
        ids = np.arange(X.shape[0])
        
        train_ids, test_ids = train_test_split(
            ids, test_size=0.2, random_state=42 + (i * i)
        )
        
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)
        
        
        model = Sequential()
        model.add(Dense(layer_dims[0],
                        activation=activation_func, input_shape=(X.shape[-1],),
                        use_bias=False)
        )
        model.add(Dropout(dropout_rate))

        for dim in layer_dims[1:]:
            model.add(Dense(dim, activation=activation_func, use_bias=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='linear',use_bias=True))
        model.compile(loss='mse', optimizer='adam')
        
        def model2actmap_fun(model):
            weights = []
            for layer in model.layers:
                if 'dense' not in layer.name:
                    continue
                weights.append(layer.get_weights()[0])

            coef = weights[0]
            for weight in weights[1:]:
                coef = np.matmul(coef,weight)
                
            return coef.ravel()
        
        best_mse = 10000
        patience_cnt = 0
        #bin_rolling_i = 0
        #bin_dict = {}
        mse_to_actmap = {(best_mse+i):None for i in range(patience)}
        
        for e in range(epochs):
            if patience_cnt > patience:
                break
                
            val_threshold = max(mse_to_actmap.keys())
            mse,act_map = _get_map(train_generator,
                                val_generator,
                                model,
                                model2actmap_fun,
                                val_threshold)
            
            if act_map is not None:
                mse_to_actmap[mse] = act_map
                del mse_to_actmap[val_threshold]

            #bin_dict[bin_rolling_i] = act_map
            #bin_rolling_i += 1
            #bin_rolling_i = bin_rolling_i % patience
            
            if mse < best_mse:
                best_mse = mse
                patience_cnt = 0
            else:
                patience_cnt += 1
            
        errors = []
        for mse, am in mse_to_actmap.items():
            coefs.append(am)
            errors.append(mse) 
            
        error = np.array(errors).mean()
        
        #for _, am in bin_dict.items():
            #coefs.append(am)
            
        if verbose > 0:
            logging.info(f'[{i+1}/{N}] - mse: {error:.04f}')

    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs
    
        
def mlp_regression(X, y,
                   save_path=None,
                   layer_dims=[1024,1024],
                   activation_func='linear',
                   dropout_rate=0.5,
                   epochs=100,
                   patience=10,
                   batch_size=64,
                   N=15,
                   verbose=0,
                   n_jobs=0):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        ids = np.arange(X.shape[0])
        #data_N = min(max_use_sample_N,X.shape[0])
        #np.random.shuffle(ids)
        #ids = ids[:data_N]
        train_ids, test_ids = train_test_split(
            ids, test_size=0.2, random_state=42 + (i * i)
        )
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)
        
        bst_model_path = f'temp_{i}_best_mlp.h5'
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
                                alpha=0.001, # mixing parameter
                                lambda_par=0.8, # shrinkage parameter
                                epochs=100,
                                patience=30,
                                batch_size=256,
                                N=15,
                                verbose=0):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        ids = np.arange(X.shape[0])
        #data_N = min(max_use_sample_N,X.shape[0])
        #np.random.shuffle(ids)
        #ids = ids[:data_N]
        
        train_ids, test_ids = train_test_split(
            ids, test_size=0.2, random_state=42 + (i * i))
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)

        bst_model_path = f'temp_{i}_best_mlp.h5'
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
               verbose=0):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []

    for i in range(N):
        ids = np.arange(X.shape[0])
        #data_N = min(max_use_sample_N,X.shape[0])
        #np.random.shuffle(ids)
        #ids = ids[:data_N]
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

