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
from scipy.stats import norm, pearsonr
from glmnet import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import datetime
from tqdm import tqdm

# TODO: replace this relative import with an absolute import.
# e.g., from {package_name}.data import loader
from ..utils.coef2map import get_map
from ..data import loader
from ..utils import config
import pdb

logging.basicConfig(level=logging.INFO)

def elasticnet(X, y,
               voxel_mask=None,
               save_path=".",
               save=True,
               alpha=0.001,
               n_repeat=5,
               n_splits=5,
               n_jobs=16,
               max_lambda=10,
               min_lambda_ratio=1e-4,
               lambda_search_num=100,
               verbose=1,
               n_samples=30000,
               confidence_interval=.99,
               n_coef_plot=150,
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
    coefs = []
    intercepts = []
    cv_mean_scores = []
    lambda_bests = []
    cv_standard_errors = []
    coef_paths = []
                    
    if verbose >0:
        iterator = tqdm(range(n_repeat))
    else:
        iterator = range(n_repeat)
        
    for i in iterator:
        np.random.seed(42+i)
        ids = np.arange(X.shape[0])

        if X.shape[0] > n_samples:
            np.random.shuffle(ids)
            ids = ids[:n_samples]

        y = y.ravel()
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
        lambda_bests.append(lambda_best) 
        
        coefs.append(coef)
        intercepts.append(intercept)
        cv_mean_scores.append(-model.cv_mean_score_)
        coef_paths.append(model.coef_path_)
        cv_standard_errors.append(model.cv_standard_error_)
        
    coefs = np.array(coefs)
    intercepts = np.array(intercepts)
    cv_mean_scores = np.array(cv_mean_scores)
    lambda_vals = np.log(np.array(lambda_bests))
    coef_paths = np.array(coef_paths)
    cv_standard_errors = np.array(cv_standard_errors)
    
    

    
    if verbose > 0:
        plot_elasticnet_result(save_root, 
                               save, 
                               coefs.mean(0), 
                               intercepts.mean(0), 
                               cv_mean_scores.mean(0), 
                               cv_standard_errors.mean(0),
                               lambda_path,
                               lambda_vals,
                               coef_paths.mean(0),
                               confidence_interval,
                               n_coef_plot)
    
    report = {"coefs": coefs,
              "intercepts": intercepts,
              "cv_mean_scores": cv_mean_scores, 
              "cv_standard_errors": cv_standard_errors,
              "lambda_path": lambda_path,
              "lambda_vals": lambda_vals,
              "coef_paths":coef_paths}
    
    if save:
        for key,data in report.items():
            np.save(save_root/f'{key}.npy', data)
        get_map(coefs, voxel_mask, task_name,
                map_type=map_type, save_path=save_root, sigma=sigma)
    
    return report

def plot_elasticnet_result(save_root, 
                           save, 
                           coefs, 
                           intercepts, 
                           cv_mean_scores, 
                           cv_standard_errors,
                           lambda_path,
                           lambda_vals,
                           coef_paths,
                           confidence_interval=.99,
                           n_coef_plot=150):
    
    
    # plot survival rate...
    plt.figure(figsize=(10, 8))
    plt.errorbar(np.log(lambda_path), cv_mean_scores,
                 yerr=cv_standard_errors* norm.ppf(1-(1-confidence_interval)/2), 
                 color='k', alpha=.5, elinewidth=1, capsize=2)
    # plot confidence interval
    plt.plot(np.log(lambda_path), cv_mean_scores, color='k', alpha=0.9)
    plt.axvspan(lambda_vals.min(), lambda_vals.max(),
                color='skyblue', alpha=.75, lw=1)
    plt.xlabel('log(lambda)', fontsize=20)
    plt.ylabel('cv average MSE', fontsize=20)
    if save:
        plt.savefig(save_root/'plot1.png',bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.plot(np.log(lambda_path), coef_paths[
             np.random.choice(np.arange(coef_paths.shape[0]), n_coef_plot), :].T)
    plt.axvspan(lambda_vals.min(), lambda_vals.max(),
                color='skyblue', alpha=.75, lw=1)
    plt.xlabel('log(lambda)', fontsize=20)
    plt.ylabel('coefficients', fontsize=20)
    if save:
        plt.savefig(save_root/'plot2.png',bbox_inches='tight')
    plt.show()
    
def elasticnet_crossvalidation(X_dict,
                               y_dict,
                               voxel_mask=None,
                               metric_function=pearsonr,
                               metric_names=['pearsonr','pvalue'],
                               method='5-fold',
                               n_cv_repeat=1,
                               cv_save=True,
                               cv_save_path=".",
                               confidence_interval=.99,
                               n_coef_plot=150,
                               task_name="unnamed",
                               map_type="z",
                               sigma=1,
                               **kwargs):
    
    if cv_save:
        now = datetime.datetime.now()
        save_root = Path(cv_save_path) / f'elasticnet_crossvalidation_report_{method}_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
        save_root.mkdir()
        
    metrics_train = []
    metrics_test = []
    reports = []
    
    def run_add_reports(X_train,y_train,X_test,y_test,metric_function,**kwargs):
        report = elasticnet(X_train,y_train, save=False,verbose=0,**kwargs)
        if len(report["coefs"].shape) == 2:
            mean_coefs = report["coefs"].mean(0)
        else:
            mean_coefs = report["coefs"]
        mean_intercepts = report["intercepts"].mean()
        pred_test = (np.matmul(mean_coefs,X_test.T) + mean_intercepts).flatten()
        pred_train = (np.matmul(mean_coefs,X_train.T) + mean_intercepts).flatten()
        metric_train = metric_function(pred_train.flatten(),y_train.flatten())
        metric_test = metric_function(pred_test.flatten(),y_test.flatten())
        
        return report, metric_train, metric_test
    
    if method=='loso':#leave-one-subject-out
        subject_list = list(X_dict.keys())
        for subject_id in tqdm(subject_list):
            X_test = X_dict[subject_id]
            y_test = y_dict[subject_id]
            X_train = np.concatenate([X_dict[v] for v in subject_list if v != subject_id],0)
            y_train = np.concatenate([y_dict[v] for v in subject_list if v != subject_id],0)
            report, metric_train, metric_test = run_add_reports(X_train,y_train,X_test,y_test,metric_function,**kwargs)
            metrics_train.append(metric_train)
            metrics_test.append(metric_test)
            reports.append(report)
            
    elif 'fold' in method:
        n_fold = int(method.split('-')[0])
        for j in tqdm(range(n_cv_repeat)):
            np.random.seed(42+j)
            ids = np.arange(X.shape[0])
            fold_size = X.shape[0]//n_fold
            for i in range(n_fold):
                test_ids = ids[fold_size*i:fold_size*(i+1)]
                train_ids = np.concatenate([ids[:fold_size*i],ids[fold_size*(i+1):]],0)
                X_test = X[test_ids]
                y_test = y[test_ids]
                X_train = X[train_ids]
                y_train = y[train_ids]
                report, metric_train, metric_test = run_add_reports(X_train,y_train,X_test,y_test,metric_function,**kwargs)
                metrics_train.append(metric_train)
                metrics_test.append(metric_test)
                reports.append(report)
        
    metrics_train = np.array(metrics_train)
    metrics_test = np.array(metrics_test)
    #coefs_train = np.array(coefs_train)
    
    
    def concat_dict(np_dict_list, key):
        
        return np.concatenate([np_dict[key] for np_dict in np_dict_list],0)
    
    plot_elasticnet_result(save_root, 
                           cv_save, 
                           concat_dict(reports,'coefs').mean(0), 
                           concat_dict(reports,'intercepts').mean(0), 
                           concat_dict(reports,'cv_mean_scores').mean(0), 
                           concat_dict(reports,'cv_standard_errors').mean(0), 
                           reports[0]['lambda_path'],
                           reports[0]['lambda_vals'],
                           concat_dict(reports,'coef_paths').mean(0), 
                           confidence_interval,
                           n_coef_plot)
    
    
    if len(metrics_train.shape) ==0:
        if metric_names is None:
            metric_name = "unnamed"
        else:
            metric_name = metric_names[0]
        plt.figure(figsize=(4, 8))
        plt.boxplot([metrics_train, metrics_test], labels=['train','test'], widths=0.6)
        if cv_save:
            plt.savefig(save_root/f'plot_{metric_name}.png',bbox_inches='tight')
        plt.show()
    else:
        for i in range(metrics_train.shape[1]):
            if metric_names is None:
                metric_name = f"unnamed_{i}"
            else:
                metric_name = metric_names[i]
            plt.figure(figsize=(4, 8))
            plt.boxplot([metrics_train[:,i], metrics_test[:,i]], labels=['train','test'], widths=0.6)
            if cv_save:
                plt.savefig(save_root/f'plot_{metric_name}.png',bbox_inches='tight')
            plt.show()
            
    if cv_save:
        for i, report in enumerate(reports):
            for key,data in report.items():
                np.save(save_root/f'{key}_{i}.npy', data)
                
        get_map(concat_dict(reports,'coefs'), voxel_mask, task_name,
                map_type=map_type, save_path=save_root, sigma=sigma)
    
    
    return metrics_train, metrics_test, reports
            
    
    