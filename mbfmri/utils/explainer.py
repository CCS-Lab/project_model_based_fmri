#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

# https://github.com/slundberg/shap/tree/9411b68e8057a6c6f3621765b89b24d82bee13d4

import numpy as np
import shap
import warnings

class Explainer():
    
    def __init__(self,
                 shap_explainer='deep',
                 shap_background_type = 'null',
                 shap_n_background = 100,
                 shap_n_sample = 1000):
    
        self.shap_explainer = shap_explainer
        self.shap_background_type = shap_background_type
        self.shap_n_background = shap_n_background
        self.shap_n_sample = shap_n_sample
        
    def _make_background(self, X_pool):
        if  self.shap_background_type == 'null':
            background = np.zeros([1]+list(X_pool.shape)[1:])
        elif self.shap_background_type == 'sample':
            background = X_pool[np.random.choice(X_pool.shape[0], 
                                                        min(X_pool.shape[0],self.shap_n_background),
                                                       replace=True)]
        elif self.shap_background_type == 'generate':
            mean = X_pool.mean(0)
            std = X_pool.std(0)
            background = np.random.randn(*([min(X_pool.shape[0],self.shap_n_background)]+list(X_pool.shape[1:])))
            background *= std
            background += mean
        else:
            raise ValueError(f'invalid shape background type {self.shap_background_type}. please choose from "null","sample" and "generate."')
        
        return background
                             
    def _deepexplainer(self,model,X_train,X_test):
        
        X_pool = np.concatenate([X_train,X_test])
        background = self._make_background(X_pool)
        sample = X_pool[np.random.choice(X_pool.shape[0], min(X_pool.shape[0],self.shap_n_sample), replace=True)]
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(sample)[0]
        return shap_values.mean(0)
    
    def _gradientexplainer(self,model,X_train,X_test):
        
        X_pool = np.concatenate([X_train,X_test])
        background = self._make_background(X_pool)
        sample = X_pool[np.random.choice(X_pool.shape[0], min(X_pool.shape[0],self.shap_n_sample), replace=True)]
        e = shap.GradientExplainer(model, background)
        shap_values = e.shap_values(sample)[0]
        return shap_values.mean(0)
    
    def _fastexplainer(self,model,X_train,X_test):
        # TODO 
        # input Eye
        pass
    
    def __call__(self,
                 model,
                X_train=None,
                X_test=None):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            if self.shap_explainer.lower() =='gradient':
                return self._gradientexplainer(model,X_train,X_test)
            elif self.shap_explainer.lower() =='deep':
                return self._deepexplainer(model,X_train,X_test)
            else:
                return self._gradientexplainer(model,X_train,X_test)
    
    