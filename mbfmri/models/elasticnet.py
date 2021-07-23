#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.03.23
    
from glmnet import ElasticNet,LogitNet
import numpy as np
from pathlib import Path
from mbfmri.models.mvpa_general import MVPA_Base, MVPA_CV
from mbfmri.utils.report import *

        
class MVPA_ElasticNet(MVPA_Base):
    
    r"""
    
    **MVPA_ElasticNet** is an MVPA model implementation of ElasticNet,
    wrapping ElasticNet from "glmnet" python package. 
    Please refer to (https://github.com/civisanalytics/python-glmnet).
    
    ElasticNet adopts a mixed L1 and L2 norm as a penalty term additional to Mean Squerred Error (MSE) in regression.
    
        - L1 norm and L2 norm is mixed as alpha * L1 + (1-alpha)/2 * L2
        - Total penalalty is modulated with shrinkage parameter : [alpha * L1 + (1-alpha)/2 * L2] * lambda
        
    Shrinkage parameter is searched through lambda search space, *lambda_path*, 
    and will be selected by comparing N-fold cross-validation MSE.
    *lambda_path* is determined by log-linearly slicing *lambda_search_num* times which exponentially decaying from *max_lambda* to *max_lambda* * *min_lambda_ratio*
    
    The model interpretation, which means extracting the weight value for each voxel, 
    is done by reading coefficient values of the linear layer.
    
    Also, additional intermediate results are reported by *report* attribute.
    The below data will be used for reporting and plotting the results.
        - 'cv_mean_score' : mean CV MSE of each CV in lambda search space
        - 'coef_path' : coefficient values of each CV in lambda search space
        - 'cv_standard_error' : SE of CV MSE of each CV in lambda search space
        - 'lambda_best' : best lambda valeu
        - 'lambda_path' : lambda search space
    
    
    
    Parameters
    ----------
    
    alpha : float, default=0.001
        Value between 0 and 1, indicating the mixing parameter in ElasticNet.
        *penalty* = [alpha * L1 + (1-alpha)/2 * L2] * lambda

    n_sample : int, default=30000
        Max number of samples used in a single fitting.
        If the number of data is bigger than *n_sample*, sampling will be done for 
        each model fitting.
        This is for preventing memory overload.

    max_lambda : float, default=10
        Maximum value of lambda in lambda search space.
        The lambda search space is used when searching the best lambda value.

    min_lambda_ratio : float, default=1e-4
        Ratio of minimum lambda value to maximum value. 
        With this ratio, a log-linearly scaled lambda space will be created.

    lambda_search_num : int, default=100
        Number of points in lambda search space. 
        Bigger the number, finer will the lambda searching be.

    n_jobs : int, default=16
        Number of cores used in fitting ElasticNet

    n_splits : int, default=5
        Number of fold used in inner cross-validation,
        which aims to find the best lambda value.
    
    logistic : bool, default=False
        Indicate if logistic regression is required.
        If True, LogitNet will be used instead, which optimizes
        logistic regression with the same penalties.
        
    """
    
    
    def __init__(self,
                 alpha=0.001,
                 n_sample=30000,
                 max_lambda=10,
                 min_lambda_ratio=1e-4,
                 lambda_search_num=100,
                 n_jobs=16,
                 n_splits=5,
                 logistic=False,
                 **kwargs):
        
        # penalty = [alpha * L1 + (1-alpha)/2 * L2] * lambda
        
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.n_sample = n_sample
        self.alpha = alpha
        self.model = None
        self.logistic = logistic
        self.lambda_path = np.exp(
                            np.linspace(
                                np.log(max_lambda),
                                np.log(max_lambda * min_lambda_ratio),
                                lambda_search_num))
        self.name = f'ElasticNet_alpha-{self.alpha}'
        
    def reset(self,**kwargs):
        if self.logistic:
            self.model = LogitNet(alpha=self.alpha,
                           n_jobs=self.n_jobs,
                           scoring='accuracy',
                           lambda_path=self.lambda_path,
                           n_splits=self.n_splits)
        else:
            self.model = ElasticNet(alpha=self.alpha,
                           n_jobs=self.n_jobs,
                           scoring='mean_squared_error',
                           lambda_path=self.lambda_path,
                           n_splits=self.n_splits)
        return
    
    def fit(self,X,y,**kwargs):
        ids = np.arange(X.shape[0])
        if X.shape[0] > self.n_sample:
            np.random.shuffle(ids)
            ids = ids[:self.n_sample]
        y = y.ravel()
        X_data = X[ids]
        y_data = y[ids]
        self.model = self.model.fit(X_data, y_data)
        return
            
    def predict(self,X,**kwargs):
        return self.model.predict(X)
        
    def get_weights(self):
        lambda_best_idx = self.model.cv_mean_score_.argmax()
        coef = self.model.coef_path_.squeeze()[:, lambda_best_idx]
        return coef
    
    def report(self,**kwargs):
        reports = {}
        reports['cv_mean_score'] = -self.model.cv_mean_score_
        reports['coef_path'] = self.model.coef_path_
        reports['cv_standard_error'] = self.model.cv_standard_error_
        lambda_best_idx = self.model.cv_mean_score_.argmax()
        reports['lambda_best'] = self.lambda_path[lambda_best_idx]
        reports['lambda_path'] = self.lambda_path
        
        return reports 
