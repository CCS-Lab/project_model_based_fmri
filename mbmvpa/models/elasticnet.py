#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.03.23
    
"""
Wrapper of ElasticNet

"""
from glmnet import ElasticNet
import numpy as np

class MVPA_ElasticNet():
    
    def __init__(self,
                 alpha=0.001,
                 n_samples=30000,
                 shuffle=True,
                 max_lambda=10,
                 min_lambda_ratio=1e-4,
                 lambda_search_num=100,
                 n_jobs=16,
                 n_splits=5):
        
        
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.n_samples = n_samples
        self.alpha = alpha
        self.model = None
        self.lambda_path = np.exp(
                            np.linspace(
                                np.log(max_lambda),
                                np.log(max_lambda * min_lambda_ratio),
                                lambda_search_num))
        
        self.name = f'ElasticNet(alpha:{self.alpha})'
        
    def reset(self):
        self.model = ElasticNet(alpha=self.alpha,
                           n_jobs=self.n_jobs,
                           scoring='mean_squared_error',
                           lambda_path=self.lambda_path,
                           n_splits=self.n_splits)
        return
    
    def fit(self,X,y):
        if self.shuffle:
            ids = np.arange(X.shape[0])
            if X.shape[0] > self.n_samples:
                np.random.shuffle(ids)
                ids = ids[:self.n_samples]
            y = y.ravel()
            X_data = X[ids]
            y_data = y[ids]
            self.model = self.model.fit(X_data, y_data)
            
        return
            
    def predict(self,X):
        return self.model.predict(X)
        
    def get_weights(self):
        lambda_best_idx = self.model.cv_mean_score_.argmax()
        lambda_best = self.lambda_path[lambda_best_idx]
        coef = self.model.coef_path_[:, lambda_best_idx]
        
        return coef
    
    def report(self):
        reports = {}
        reports['cv_mean_score'] = -self.model.cv_mean_score_
        reports['coef_path'] = self.model.coef_path_
        reports['cv_standard_error'] = self.model.cv_standard_error_
        lambda_best_idx = self.model.cv_mean_score_.argmax()
        reports['lambda_best'] = self.lambda_path[lambda_best_idx]
        reports['lambda_path'] = self.lambda_path
        
        return reports 
    
    