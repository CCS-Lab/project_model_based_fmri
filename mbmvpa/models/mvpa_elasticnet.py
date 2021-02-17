#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho, Yedarm Seong
#contact: cjfwndnsl@gmail.com, mybirth0407@gmail.com
#last modification: 2020.11.16
    
"""
"""

import logging
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from glmnet import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime

# TODO: replace this relative import with an absolute import.
# e.g., from {package_name}.data import loader
from ..utils.coef2map import get_map
from ..data import loader
from ..utils import config


logging.basicConfig(level=logging.INFO)

def elasticnet(X, y,
               voxel_mask,
               save_path=".",
               save=True,
               alpha=0.001,
               n_splits=5,
               n_jobs=16,
               max_lambda=10,
               min_lambda_ratio=1e-4,
               lambda_search_num=100,
               verbose=1,
               n_samples=30000,
               confidence_interval=.99,
               task_name="unnamed",
               map_type="z",
               sigma=1
               ):
    """
    This package is wrapping ElasticNet from "glmnet" python package. please refer to (https://github.com/civisanalytics/python-glmnet)
    Fitting ElasticNet as a regression model for Multi-Voxel Pattern Analysis and extracting fitted coefficients.
    L1 norm and L2 norm is mixed as alpha * L1 + (1-alpha)/2 * L2
    Total penalalty is modulated with shrinkage parameter : [alpha * L1 + (1-alpha)/2 * L2] * lambda
    Shrinkage parameter is searched through "lambda_path" calculating N fold (=n_splits) cross-validation for each.
    "lambda_path" is determined by linearly slicing "lambda_search_num" times which exponentially decaying from "max_lambda" to "max_lambda" * "min_lambda_ratio"
    
    Repeat several times (=N) and return N coefficients.

    Args:
        X (numpy.ndarray): preprocessed fMRI data. shape : data # x voxel #
        y (numpy.ndarray): parametric modulation values to regress X against. shape: data #
        alpha (float): mixing parameter
        n_splits (int): the number of N-fold cross validation
        n_jobs (int): the number of cores for parallel computing
        max_lambda (float): the maximum value of lambda to search
        min_lambda_ratio (float): the ratio of minimum lambda value to maximum lambda value. 
        lambda_search_num (int): the number of searching candidate.
        verbose (int): if > 0 then log fitting process and report a validation mse of each repetition.
        save (bool): if True save the results
        save_path (str or Path): save temporal model weights file. TODO : replace it with using invisible temp file
        n_samples (int): maximum number of instance of data (X,y) used in a single repetition. 
        confidence_interval (float): confidence interval for plotting fitting results. default is .99 for 99% confidence interval.

    Return:
        numpy.ndarray : **coefs** (*numpy.array*) - fitted models' coefficients mapped to weight of each voxel.  shape: N x voxel #.
    """
    
    if save:
        now = datetime.datetime.now()
        save_root = Path(save_path) / f'elasticnet_report_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
        save_root.mkdir()
    
    exponent = np.linspace(
        np.log(max_lambda),
        np.log(max_lambda * min_lambda_ratio),
        lambda_search_num)

    # making lambda candidate list for searching best lambda
    lambda_path = np.exp(exponent)

        
    # random sampling "n_samples" if the given number of X,y instances is bigger
    np.random.seed(42)
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
    intercept = model.intercept_path_[lambda_best_idx]
    lambda_vals = np.log(np.array([lambda_best]))
    coefs = np.array([coef])
    intercepts = np.array([intercept])
    if save:
        np.save(save_root/'cv_mean_score.npy', -model.cv_mean_score_)
        np.save(save_root/'coef.npy',model.coef_path_)
        np.save(save_root/'intercept.npy',model.intercept_path_)
        get_map(coefs, voxel_mask, task_name,
                map_type=map_type, save_path=save_root, sigma=sigma)

    if verbose > 0:

        # visualization of ElasticNet procedure
        print(f'- lambda_best: {lambda_best:.03f}, mse: {error:.04f}, survival rate (non-zero): {(abs(coefs)>0).sum()}/{coefs.flatten().shape[0]}')
        plt.figure(figsize=(10, 8))
        plt.errorbar(np.log(lambda_path), -model.cv_mean_score_,
                     yerr=model.cv_standard_error_* norm.ppf(1-(1-confidence_interval)/2), 
                     color='k', alpha=.5, elinewidth=1, capsize=2)
        # plot confidence interval
        plt.plot(np.log(lambda_path), -
                 model.cv_mean_score_, color='k', alpha=0.9)
        plt.axvspan(lambda_vals.min(), lambda_vals.max(),
                    color='skyblue', alpha=0.2, lw=1)
        plt.xlabel('log(lambda)', fontsize=20)
        plt.ylabel('cv average MSE', fontsize=20)
        if save:
            plt.savefig(save_root/'plot1.png',bbox_inches='tight')
        plt.show()
        plt.figure(figsize=(10, 8))
        plt.plot(np.log(lambda_path), model.coef_path_[
                 np.random.choice(np.arange(model.coef_path_.shape[0]), 150), :].T)
        plt.axvspan(lambda_vals.min(), lambda_vals.max(),
                    color='skyblue', alpha=.75, lw=1)
        plt.xlabel('log(lambda)', fontsize=20)
        plt.ylabel('coefficients', fontsize=20)
        if save:
            plt.savefig(save_root/'plot2.png',bbox_inches='tight')
        plt.show()

    # coefs : N x voxel #
    return coefs, intercepts
