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
from pathlib import Path
from mbmvpa.models.mvpa_general import MVPA_Base, MVPA_CV
from mbmvpa.utils.report import build_elasticnet_report_functions


class MVPACV_ElasticNet(MVPA_CV):
    
    def __init__(self,
                 X_dict,
                 y_dict,
                 voxel_mask,
                 method='5-fold',
                 n_cv_repeat=1,
                 cv_save=True,
                 cv_save_path=".",
                 experiment_name="unnamed",
                 alpha=0.001,
                 n_samples=30000,
                 shuffle=True,
                 max_lambda=10,
                 min_lambda_ratio=1e-4,
                 lambda_search_num=100,
                 n_jobs=16,
                 n_splits=5,
                 confidence_interval=.99,
                 n_coef_plot=150,
                 map_type='z',
                 sigma=1):
    
        self.model = MVPA_ElasticNet(alpha=alpha,
                                    n_samples=n_samples,
                                    shuffle=shuffle,
                                    max_lambda=max_lambda,
                                    min_lambda_ratio=min_lambda_ratio,
                                    lambda_search_num=lambda_search_num,
                                    n_jobs=n_jobs,
                                    n_splits=n_splits)

        self.report_function_dict = build_elasticnet_report_functions(voxel_mask=voxel_mask,
                                                                     confidence_interval=confidence_interval,
                                                                     n_coef_plot=n_coef_plot,
                                                                     experiment_name=experiment_name,
                                                                     map_type=map_type,
                                                                     sigma=sigma)

        super().__init__(X_dict=X_dict,
                        y_dict=y_dict,
                        model=self.model,
                        method=method,
                        n_cv_repeat=n_cv_repeat,
                        cv_save=cv_save,
                        cv_save_path=cv_save_path,
                        experiment_name=experiment_name,
                        report_function_dict=self.report_function_dict)
    
    
class MVPA_ElasticNet(MVPA_Base):
    
    def __init__(self,
                 alpha=0.001,
                 n_samples=30000,
                 shuffle=True,
                 max_lambda=10,
                 min_lambda_ratio=1e-4,
                 lambda_search_num=100,
                 n_jobs=16,
                 n_splits=5,
                 **kwargs):
        
        # penalty = [alpha * L1 + (1-alpha)/2 * L2] * lambda
        
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
        
    def reset(self,**kwargs):
        self.model = ElasticNet(alpha=self.alpha,
                           n_jobs=self.n_jobs,
                           scoring='mean_squared_error',
                           lambda_path=self.lambda_path,
                           n_splits=self.n_splits)
        return
    
    def fit(self,X,y,**kwargs):
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
            
    def predict(self,X,**kwargs):
        return self.model.predict(X)
        
    def get_weights(self):
        lambda_best_idx = self.model.cv_mean_score_.argmax()
        lambda_best = self.lambda_path[lambda_best_idx]
        coef = self.model.coef_path_[:, lambda_best_idx]
        
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
