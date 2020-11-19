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
import random
import os

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from scipy.stats import ttest_1samp
from ..data import loader
from glmnet import ElasticNet
import matplotlib.pyplot as plt

import logging

logging.basicConfig(level=logging.INFO)


def mlp_regression(X, y, # input data
                   # specification for model design & training
                   layer_dims=[1024, 1024],
                   activation="linear",
                   activation_output="linear",
                   dropout_rate=0.5,
                   epochs=100,
                   patience=10,
                   batch_size=64,
                   validation_split=0.2,
                   N=15,
                   verbose=0,
                   optimizer="adam",
                   loss="mse",
                   save=True,
                   save_path=None,
                   n_samples=30000):
    
    if verbose > 0:
        logging.info("start running")

    coefs = []

    for i in range(1, N + 1):
        np.random.seed(i * i)
        tf.random.set_seed(i *i)
        ids = np.arange(X.shape[0])

        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]

        train_ids, test_ids = train_test_split(
            ids, test_size=validation_split, random_state=(i * i)
        )
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)
        
        if save_path is None:
            sp = Path(layout.derivatives["fMRIPrep"].root) / "data"
        else:
            sp = Path(save_path)
        
        if save and not sp.exists():
            sp.mkdir()

        sp / "model_ckpt".mkdir()
        best_model_filepath = sp / f"model_ckpt/cp-{i}-{epoch:03d}.ckpt"
        
        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min"
        )
        es = EarlyStopping(monitor="val_loss", patience=patience)
        
        model = Sequential()
        model.add(Dense(layer_dims[0],
                        activation=activation,
                        input_shape=(X.shape[-1],),
                        use_bias=False)
        )
        model.add(Dropout(dropout_rate))
        
        for dim in layer_dims[1:]:
            model.add(Dense(dim, activation=activation, use_bias=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation=activation_output, use_bias=True))
        model.compile(loss=loss, optimizer=optimizer)

        model.fit(train_generator,
            batch_size=batch_size, epochs=epochs,
            verbose=0, callbacks=[mc, es],
            validation_data=val_generator,
            steps_per_epoch=train_steps,
            validation_steps=val_steps
        )

        model.load_weights(best_model_filepath)

        # results = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        error = mean_squared_error(y_pred, y_test)

        if verbose > 0:
            logging.info(f"[{i}/{N}] - {loss}: {error:.04f}")

        weights = []
        for layer in model.layers:
            if "dense" not in layer.name:
                continue
            weights.append(layer.get_weights()[0])

        coef = weights[0]
        for weight in weights[1:]:
            coef = np.matmul(coef,weight)
        coefs.append(coef.ravel())

    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs


def penalized_linear_regression(X, y, # input data
                                activation="linear",
                                activation_output="linear",
                                alpha=0.001, # mixing parameter
                                lambda_par=0.8, # shrinkage parameter
                                epochs=100,
                                patience=30,
                                batch_size=64,
                                validation_split=0.2,
                                N=15,
                                verbose=0,
                                optimizer="adam",
                                loss="mse",
                                save=True,
                                save_path=None,
                                n_samples=30000
                                ):

    if verbose > 0:
        logging.info("start running")
        
    coefs = []

    for i in range(1, N + 1):
        np.random.seed(i * i)
        tf.random.set_seed(i *i)

        ids = np.arange(X.shape[0])
        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]
        
        train_ids, test_ids = train_test_split(
            ids, test_size=validation_split, random_state=42 + (i * i))
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(X_test, y_test, batch_size, shuffle=False)

        if save_path is None:
            sp = Path(layout.derivatives["fMRIPrep"].root) / "data"
        else:
            sp = Path(save_path)

        if save and not sp.exists():
            sp.mkdir()

        sp / "model_ckpt".mkdir()
        best_model_filepath = sp / f"model_ckpt/cp-{i}-{epoch:03d}.ckpt"
        
        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min"
        )

        es = EarlyStopping(monitor="val_loss",
                patience=patience,
                save_best_only=True,
                save_weights_only=True,
                mode="min"
            )

        kernel_regularizer = l1_l2(lambda_par * alpha, lambda_par * (1 - alpha) / 2)

        model = Sequential()
        model.add(Dense(1, activation=activation, input_shape=(X.shape[-1]),
                        use_bias=True, kernel_regularizer=kernel_regularizer))
        model.compile(loss=loss, optimizer=optimizer)

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
            logging.info(f'[{i}/{N}] - mse: {error:.04f}')

        coef = model.layers[0].get_weights()[0] 
        coefs.append(coef.ravel())
        os.remove(bst_model_path)
        
    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs


def elasticnet(X, y,
               alpha=0.001,
               n_splits=5,
               n_jobs=16,
               max_lambda=10,
               min_lambda_ratio=1e-4,
               lambda_search_num=100,
               N=1,
               verbose=0,
               n_samples=30000):
    
    if verbose > 0:
        logging.info('start running')
        
    coefs = []
    
    exponent = np.linspace(
                    np.log(max_lambda),
                    np.log(max_lambda * min_lambda_ratio),
                    lambda_search_num
                )

    lambda_path = np.exp(exponent)

    for i in range(1, N + 1):
        np.random.seed(i * i)

        ids = np.arange(X.shape[0])
        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]

        X_data = X[ids]
        y_data  = y[ids]
        model = ElasticNet(alpha=alpha,
                    n_jobs =n_jobs,
                    scoring='mean_squared_error',
                    lambda_path=lambda_path,
                    n_splits=n_splits
                )

        model = model.fit(X_data,y_data)
        y_pred = model.predict(X_data).flatten()
        error = mean_squared_error(y_pred, y_data)
        
        lambda_best_idx = model.cv_mean_score_.argmax()
        lambda_best = lambda_path[lambda_best_idx]
        coef = model.coef_path_[:,lambda_best_idx]
        lambda_vals= np.log(np.array([lambda_best]))
        coefs.append(coef)
        
        if verbose > 0:
            logging.info(f'[{i}/{N}] - lambda_best: {lambda_best:.03f}/ mse: {error:.04f}')
            plt.figure(figsize=(10,8))
            plt.errorbar(np.log(lambda_path),-model.cv_mean_score_,yerr=model.cv_standard_error_*2.576,color='k',alpha=.5,elinewidth=1,capsize=2)
            # plot 99 % confidence interval
            plt.plot(np.log(lambda_path),-model.cv_mean_score_, color='k', alpha = 0.9)
            plt.axvspan(lambda_vals.min(), lambda_vals.max(), color='skyblue', alpha=0.2, lw=1)
            plt.xlabel('log(lambda)',fontsize=20)
            plt.ylabel('cv average MSE',fontsize=20)
            plt.show()
            plt.figure(figsize=(10,8))
            plt.plot(np.log(lambda_path),model.coef_path_[np.random.choice(np.arange(model.coef_path_.shape[0]),150),:].T)
            plt.axvspan(lambda_vals.min(), lambda_vals.max(), color='skyblue', alpha=.75, lw=1)
            plt.xlabel('log(lambda)',fontsize=20)
            plt.ylabel('coefficients',fontsize=20)
            plt.show()
            
    
    coefs = np.array(coefs)
    
    # coefs : N x voxel #
    return coefs

