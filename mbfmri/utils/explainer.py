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
        # sample : (n_sample, *X_shape)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            if self.shap_explainer.lower() =='gradient':
                e = shap.GradientExplainer(model, background)
            elif self.shap_explainer.lower() =='deep':
                e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(sample)[0]
            
        output = {'shap_values':shap_values,
                 'shap_sample':sample,
                 'shap_pred':preds[sample_idxs]}
        
        return output
    
        '''
        non_zero_mask = (abs(sample.mean(0)) > 1e-8)
        original_shape = shap_values.shape[1:]
        transpose_index = list(range(len(shap_values.shape)))
        transpose_index = transpose_index[1:] + transpose_index[:1]
        shap_values=shap_values.transpose(*transpose_index)[non_zero_mask].T
        sample=sample.transpose(*transpose_index)[non_zero_mask].T
        regs = [linregress(sample[:,i],shap_values[:,i]) for i in range(sample.shape[1])]
        pvalue = [reg.pvalue for reg in regs]
        valid,_ = fdrcorrection(pvalue, alpha=self.pval_threshold)
        slopes = np.array([reg.slope for reg in regs])
        slopes[~valid] = 0
        blackboard = np.zeros(original_shape)
        blackboard[non_zero_mask] = slopes
        slopes = blackboard

        return slopes
        '''
        