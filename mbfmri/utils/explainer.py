#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

# https://github.com/slundberg/shap/tree/9411b68e8057a6c6f3621765b89b24d82bee13d4

import numpy as np
import shap

class Explainer():
    
    def __init__(self,
                 shap_explainer='deep',
                 shap_null_background = False,
                 shap_n_background = 100,
                 shap_n_sample = 1000):
    
        self.shap_explainer = shap_explainer
        self.shap_null_background = shap_null_background
        self.shap_n_background = shap_n_background
        self.shap_n_sample = shap_n_sample
        
    def _deepexplainer(self,model,X_train,X_test):
        
        if  self.shap_null_background:
            background = np.zeros([1]+list(X_train.shape)[1:])
        else:
            background = X_train[np.random.choice(X_train.shape[0], 
                                                        min(X_train.shape[0],self.shap_n_background),
                                                       replace=True)]
        sample = X_test[np.random.choice(X_test.shape[0], min(X_test.shape[0],self.shap_n_sample), replace=True)]
        e = shap.DeepExplainer(model, background)
        shap_values = e.shap_values(sample)[0]
        return shap_values.mean(0)
    
    def _gradientexplainer(self,model,X_train,X_test):
        
        if  self.shap_null_background:
            background = np.zeros([1]+list(X_train.shape)[1:])
        else:
            background = X_train[np.random.choice(X_train.shape[0], 
                                                       min(X_train.shape[0],self.shap_n_background), 
                                                       replace=True)]
        sample = X_test[np.random.choice(X_test.shape[0], min(X_test.shape[0],self.shap_n_sample), replace=True)]
        e = shap.GradientExplainer(model,background)
        shap_values = e.shap_values(sample)[0]
        return shap_values.mean(0)
    
    def __call__(self,
                 model,
                X_train=None,
                X_test=None):
        
        if self.shap_explainer.lower() =='deep':
            return self._gradientexplainer(model,X_train,X_test)
        elif self.shap_explainer.lower() =='perturbation':
            return self._deepexplainer(model,X_train,X_test)
        else:
            return self._gradientexplainer(model,X_train,X_test)
    
    