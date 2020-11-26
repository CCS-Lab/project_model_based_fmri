#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Cheol Jun Cho, Yedarm Seong
@contact: cjfwndnsl@gmail.com
          mybirth0407@gmail.com
@last modification: 2020.11.16


This part is implemented to fit regression model and extract voxel-wise weights (coefficients). 

Available Model:
    Multi-Layer Perceptron (tf.Keras): stacked perceptron layers intertwinned with "Drop out".
    Penalized linear regression (Keras): penalizing obejctive function with mixed L1 and L2 norm.
    ElasticNet (glmnet): penalized linear regression with automatical searching optimal amount of penalizing (shrinkage parameter).
"""

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from glmnet import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2

# TODO: replace this relative import with an absolute import.
# e.g., from {package_name}.data import loader
from ..data import loader

DEFAULT_SAVE_PATH_TEMP = 'temp'
logging.basicConfig(level=logging.INFO)


def mlp_regression(X, y,
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
                   layout=None,
                   save=False,
                   save_path=None,
                   n_samples=30000):
    """
    Fitting Multi-Layer Perceptron (MLP) as a regression model for multi-voxel
    pattern analysis and extracting fitted coefficients. Mini-batch fitting
    with earlystopping using gradient descent (Adam).

    Coefficient extraction is done by sequential matrix multiplication of
    layers. The activation function is assumed to be linear.
    Repeat several times (=N) and return N coefficients.

    Arguments:
        -- Data --
        X : flattened fMRI data. shape : data # x voxel #
        y : parametric modulation values to regress X against. shape: data #

        -- Model hyperparameters--
        layer_dims: specification of # of hidden neurons in each linear layer. list(# of ith layer hidden dimension)
                    MLP will be constructed by stacking layers with specified hidden dimension.
        activation: activation function applied after each layer.
                     e.g. 'linear' : f(x) = x, 'sigmoid' : f(x) = 1 / (1 + exp(-x))
                     any other activation functions defined in Keras can be used.  (https://www.tensorflow.org/api_docs/python/tf/keras/activations)
        activation_output: activation function applied after final layer. should reflect the nature of y. e.g. regression on y : use 'linear'
        dropout_rate: drop out rate for drop out layers intertwinned between consecutive linear layers.

        epochs: maximum number of iterations
        batch_size: number of instance used in a single mini-batch.
        patience: parameter for early stopping, indicating the maximum number for patiently seeking better model.
                  if the better fitting validation performance is not reached within patience number, the fitting will end.
        validation_split: the ratio of validation set.
                          e.g. 0.2 means 20 % of given data will be used for validation and 80 % will be used for training
        optimizer: optimizer used for fitting model. default: 'Adam'
                   please refer to Keras optimizer api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
        loss: objective function to minimize in training. as it is a regression, default is 'mse' (Mean Squared Error)
              please refer to Keras loss api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/losses)

        -- Others --
        verbose: if > 0 then log fitting process and report a validation mse of each repitition.
        save: if True save fitted weigths, else erase them.
        save_path: save temporal model weights file. TODO : replace it with using invisible temp file
        n_samples: maximum number of instance of data (X,y) used in a single repetition. 

    Return:
        coefs: N fitted models' coefficients mapped to weight of each voxel.  shape: N x voxel #. 
    """

    if verbose > 0:
        logging.info("start running")

    coefs = []

    for i in range(1, N + 1):
        # random sampling "n_samples" if the given number of X,y instances is bigger
        np.random.seed(i * i)
        tf.random.set_seed(i * i)
        ids = np.arange(X.shape[0])

        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]

        # general fitting framework using Keras
        train_ids, test_ids = train_test_split(
            ids, test_size=validation_split, random_state=(i * i)
        )
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(
            X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(
            X_test, y_test, batch_size, shuffle=False)

        if save_path is None:
            sp = Path(
                layout.derivatives["fMRIPrep"].root) / DEFAULT_SAVE_PATH_TEMP
        else:
            sp = Path(save_path)

        best_model_filepath = sp / \
            f'temp{int(random.random()*100000)}_best_mlp.h5'

        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min")
        es = EarlyStopping(monitor="val_loss", patience=patience)

        # MLP model building
        model = Sequential()
        model.add(Dense(layer_dims[0],
                        activation=activation,
                        input_shape=(X.shape[-1],),
                        use_bias=False))
        model.add(Dropout(dropout_rate))

        for dim in layer_dims[1:]:
            model.add(Dense(dim, activation=activation, use_bias=False))
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation=activation_output, use_bias=True))
        model.compile(loss=loss, optimizer=optimizer)

        # model fitting
        model.fit(train_generator,
                  batch_size=batch_size, epochs=epochs,
                  verbose=0, callbacks=[mc, es],
                  validation_data=val_generator,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps)

        # load best model
        model.load_weights(best_model_filepath)

        # results = model.evaluate(X_test, y_test)
        y_pred = model.predict(X_test)
        error = mean_squared_error(y_pred, y_test)
        actual_epoch = len(model.history['val_loss'])
        if verbose > 0:
            logging.info(
                f"[{i}/{N}] - val_{loss}: {error:.04f}, epoch:{actual_epoch}")

        # extracting voxel-wise mapped weight (coefficient) map
        weights = []
        for layer in model.layers:
            if "dense" not in layer.name:
                continue
            weights.append(layer.get_weights()[0])

        coef = weights[0]
        for weight in weights[1:]:
            coef = np.matmul(coef, weight)
        coefs.append(coef.ravel())

    coefs = np.array(coefs)

    # coefs : N x voxel #
    return coefs


def penalized_linear_regression(X, y,
                                alpha=0.001,
                                lambda_param=0.8,
                                epochs=100,
                                patience=30,
                                batch_size=64,
                                validation_split=0.2,
                                N=15,
                                verbose=0,
                                optimizer="adam",
                                loss="mse",
                                layout=None,
                                save=False,
                                save_path=None,
                                n_samples=30000):
    """
    Fitting penalized linear regression model as a regression model for Multi-Voxel Pattern Analysis and extracting fitted coefficients.
    L1 norm and L2 norm is mixed as alpha * L1 + (1-alpha)/2 * L2
    Total penalalty is modulated with shrinkage parameter: [alpha * L1 + (1-alpha)/2 * L2] * lambda
    Mini-batch fitting with earlystopping using gradient descent (Adam).
    Coefficient extraction is done by sequential matrix multiplication of layers.
    The activation function is assumed to be linear.
    Repeat several times (=N) and return N coefficients.

    Arguments:
        -- Data --
        X: flattened fMRI data. shape : data # x voxel #
        y: parametric modulation values to regress X against. shape: data #

        -- Model hyperparameters --
        alpha : mixing parameter
        lambda_param : shrinkage parameter
        epochs : maximum number of iterations.
        batch_size : number of instance used in a single mini-batch.
        patience : parameter for early stopping, indicating the maximum number for patiently seeking better model.
                   if the better fitting validation performance is not reached within patience number, the fitting will end.
        validation_split : the ratio of validation set. 
                           e.g. 0.2 means 20 % of given data will be used for validation and 80 % will be used for training
        optimizer : optimizer used for fitting model. default: 'Adam'
                    please refer to Keras optimizer api to use another. (https://keras.io/api/optimizers/)
        loss : objective function to minimize in training. as it is a regression, default is 'mse' (Mean Squared Error)
               please refer to Keras loss api to use another. (https://keras.io/api/losses/)

        -- Others --
        verbose: if > 0 then log fitting process and report a validation mse of each repitition.
        save: if True save the results
        save_path: save temporal model weights file. TODO : replace it with using invisible temp file
        n_samples: maximum number of instance of data (X,y) used in a single repetition.

    Return:
        coefs: N fitted models' coefficients mapped to weight of each voxel.  shape: N x voxel #.
    """

    logging.info("start running")

    coefs = []

    for i in range(1, N + 1):

        # random sampling "n_samples" if the given number of X,y instances is bigger
        np.random.seed(i * i)
        tf.random.set_seed(i * i)

        ids = np.arange(X.shape[0])
        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]

        # general fitting framework using Keras
        train_ids, test_ids = train_test_split(
            ids, test_size=validation_split, random_state=42 + (i * i))
        train_steps = len(train_ids) // batch_size
        val_steps = len(test_ids) // batch_size

        X_train = X[train_ids]
        X_test = X[test_ids]
        y_train = y[train_ids]
        y_test = y[test_ids]

        train_generator = loader.DataGenerator(
            X_train, y_train, batch_size, shuffle=True)
        val_generator = loader.DataGenerator(
            X_test, y_test, batch_size, shuffle=False)

        if save_path is None:
            sp = Path(
                layout.derivatives["fMRIPrep"].root) / DEFAULT_SAVE_PATH_TEMP
        else:
            sp = Path(save_path)

        best_model_filepath = sp / \
            f'temp{int(random.random()*100000)}_best_mlp.h5'

        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min")

        es = EarlyStopping(monitor="val_loss",
                           patience=patience,
                           save_best_only=True,
                           save_weights_only=True,
                           mode="min")

        # penalizing
        kernel_regularizer = l1_l2(
            lambda_param * alpha, lambda_param * (1 - alpha) / 2)

        # model building
        model = Sequential()
        model.add(Dense(1, activation="linear", input_shape=(X.shape[-1]),
                        use_bias=True, kernel_regularizer=kernel_regularizer))
        model.compile(loss=loss, optimizer=optimizer)

        # model fitting
        model.fit(train_generator,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=0,
                  callbacks=[mc, es],
                  validation_data=val_generator,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps)

        model.load_weights(best_model_filepath)

        y_pred = model.predict(X_test)
        error = mean_squared_error(y_pred, y_test)
        actual_epoch = len(model.history['val_loss'])

        if verbose > 0:
            logging.info(
                f"[{i}/{N}] - val_{loss}: {error:.04f}, epoch:{actual_epoch}")

        # extracting coefficients
        coef = model.layers[0].get_weights()[0]
        coefs.append(coef.ravel())

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
    """
    This package is wrapping ElasticNet from "glmnet" python package. please refer to (https://github.com/civisanalytics/python-glmnet)
    Fitting ElasticNet as a regression model for Multi-Voxel Pattern Analysis and extracting fitted coefficients.
    L1 norm and L2 norm is mixed as alpha * L1 + (1-alpha)/2 * L2
    Total penalalty is modulated with shrinkage parameter : [alpha * L1 + (1-alpha)/2 * L2] * lambda
    Shrinkage parameter is searched through "lambda_path" calculating N fold (=n_splits) cross-validation for each.
    "lambda_path" is determined by linearly slicing "lambda_search_num" times which exponentially decaying from "max_lambda" to "max_lambda" * "min_lambda_ratio"
    Mini-batch fitting with earlystopping using gradient descent (Adam).
    Coefficient extraction is done by sequential matrix multiplication of layers. The activation function is assumed to be linear.
    Repeat several times (=N) and return N coefficients.

    Arugments:
        -- Data --
        X: flattened fMRI data. shape : data # x voxel #
        y: parametric modulation values to regress X against. shape: data #

        -- Model hyperparameters --
        alpha: mixing parameter
        n_splits : the number of N-fold cross validation
        n_jobs : the number of cores for parallel computing
        max_lambda : the maximum value of lambda to search
        min_lambda_ratio : the ratio of minimum lambda value to maximum lambda value. 
        lambda_search_num : the number of searching candidate.

        -- Others --
        verbose : if > 0 then log fitting process and report a validation mse of each repitition.
        save : if True save the results
        save_path : save temporal model weights file. TODO : replace it with using invisible temp file
        n_samples : maximum number of instance of data (X,y) used in a single repetition. 

    Return:
        coefs : N fitted models' coefficients mapped to weight of each voxel.
                shape: N x voxel #. 
    """

    logging.info('start running')
    coefs = []
    exponent = np.linspace(
        np.log(max_lambda),
        np.log(max_lambda * min_lambda_ratio),
        lambda_search_num)

    # making lambda candidate list for searching best lambda
    lambda_path = np.exp(exponent)

    for i in range(1, N + 1):
        # random sampling "n_samples" if the given number of X,y instances is bigger
        np.random.seed(i * i)
        ids = np.arange(X.shape[0])

        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]

        X_data = X[ids]
        y_data = y[ids]

        # ElasticNet by glmnet package
        model = ElasticNet(alpha=alpha,
                           n_jobs=n_jobs,
                           scoring='mean_squared_error',
                           lambda_path=lambda_path,
                           n_splits=n_splits)

        model = model.fit(X_data, y_data)
        y_pred = model.predict(X_data).flatten()
        error = mean_squared_error(y_pred, y_data)

        lambda_best_idx = model.cv_mean_score_.argmax()
        lambda_best = lambda_path[lambda_best_idx]

        # extracting coefficients
        coef = model.coef_path_[:, lambda_best_idx]
        lambda_vals = np.log(np.array([lambda_best]))
        coefs.append(coef)
        coefs = np.array(coefs)

        if verbose > 0:
            # visualization of ElasticNet procedure
            logging.info(
                f'[{i}/{N}] - lambda_best: {lambda_best:.03f}/ mse: {error:.04f}')
            plt.figure(figsize=(10, 8))
            plt.errorbar(np.log(lambda_path), -model.cv_mean_score_,
                         yerr=model.cv_standard_error_*2.576, color='k', alpha=.5, elinewidth=1, capsize=2)
            # plot 99 % confidence interval
            plt.plot(np.log(lambda_path), -
                     model.cv_mean_score_, color='k', alpha=0.9)
            plt.axvspan(lambda_vals.min(), lambda_vals.max(),
                        color='skyblue', alpha=0.2, lw=1)
            plt.xlabel('log(lambda)', fontsize=20)
            plt.ylabel('cv average MSE', fontsize=20)
            plt.show()
            plt.figure(figsize=(10, 8))
            plt.plot(np.log(lambda_path), model.coef_path_[
                     np.random.choice(np.arange(model.coef_path_.shape[0]), 150), :].T)
            plt.axvspan(lambda_vals.min(), lambda_vals.max(),
                        color='skyblue', alpha=.75, lw=1)
            plt.xlabel('log(lambda)', fontsize=20)
            plt.ylabel('coefficients', fontsize=20)
            plt.show()

    # coefs : N x voxel #
    return coefs
