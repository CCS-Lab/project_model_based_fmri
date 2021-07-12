#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

# https://github.com/slundberg/shap/tree/9411b68e8057a6c6f3621765b89b24d82bee13d4

import numpy as np
import shap
import warnings
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow as tf
from scipy.stats import linregress
from statsmodels.stats.multitest import fdrcorrection
tf.compat.v1.disable_v2_behavior()
#shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


class Explainer():
    
    def __init__(self,
                 shap_explainer='deep',
                 shap_n_background = 100,
                 shap_n_sample = 100,
                 include_trainset=True,
                 pval_threshold=0.05,
                 n_bin=10):
    
        self.shap_explainer = shap_explainer
        self.shap_n_background = shap_n_background
        self.shap_n_sample = shap_n_sample
        self.include_trainset = include_trainset
        self.pval_threshold = pval_threshold
        self.n_bin = n_bin
    '''
    def _balanced_sample(self, arr,n_bin,n_sample):
        std = arr.std()
        mean = arr.mean()
        ub = min(mean+ 3*std,arr.max())
        lb = max(mean- 3*std,arr.min())
        temp = arr.copy()
        temp[temp>=ub] = ub
        temp[temp<=lb] = lb
        bin_size = (ub-lb)/n_bin
        bin_arr = temp//bin_size
        bin_arr[bin_arr==(ub//bin_size)] = (ub//bin_size)-1
        bin_labels = np.arange(lb//bin_size,ub//bin_size)
        bin_pools = {l:np.nonzero(bin_arr==l)[0] for l in bin_labels}
        sample = []
        for i in range(n_sample):
            pool = bin_pools[bin_labels[i%n_bin]]
            if len(pool) == 0:
                continue
            sample.append(np.random.choice(pool))
        return np.array(sample)
    '''
    
    
    def __call__(self,model,X_test,X_train):
        if self.include_trainset:
            X_test = np.concatenate([X_test,X_train])
        preds = model.predict(X_test)
        preds = preds.ravel()
        X_bg = X_test
        X_samp = X_test
        background =  X_bg[np.random.choice(len(X_bg),self.shap_n_background)]
        #sample_idxs =  self._balanced_sample(preds,self.n_bin,self.shap_n_sample)
        sample_idxs = np.random.choice(len(X_samp),self.shap_n_sample)
        sample = X_samp[sample_idxs]
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            if self.shap_explainer.lower() =='gradient':
                e = shap.GradientExplainer(model, background)
            elif self.shap_explainer.lower() =='deep':
                e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(sample)[0]
        
        
        # get r for
        rvalue = []
        pvalue = []
        
        shap_preds = preds[sample_idxs]
        for i in range(shap_values.shape[1]):
            reg = linregress(shap_values[:,i],shap_preds)
            rvalue.append(reg.rvalue)
            pvalue.append(reg.pvalue)
        
        rvalue = np.array(rvalue)
        pvalue = np.array(pvalue)
        
        # fdr correction
        rejected,corrected = fdrcorrection(pvalue, alpha=self.pval_threshold)
        rvalue[~rejected] = 0
        
        return rvalue
    