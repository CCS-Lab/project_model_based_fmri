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
tf.compat.v1.disable_v2_behavior()
#shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough


class Explainer():
    
    def __init__(self,
                 shap_explainer='deep',
                 shap_n_background = 100,
                 shap_n_sample = 100,
                 shap_bg_range=[0,.2],
                 shap_samp_range=[.8,1],
                 shap_use_ratio=True,
                 include_trainset=False):
    
        self.shap_explainer = shap_explainer
        self.shap_n_background = shap_n_background
        self.shap_n_sample = shap_n_sample
        
        self.shap_bg_range = shap_bg_range
        self.shap_samp_range = shap_samp_range
        self.shap_use_ratio = shap_use_ratio
        self.include_trainset = include_trainset
                             
    def _get_thresholds(self, arr):
        if self.shap_use_ratio:
            
            def clip(x,a_max,a_min):
                return min(a_max,max(a_min,x))
            
            temp = arr.copy().ravel()
            temp.sort()
            l = len(temp)
            lo_bg = temp[clip(int(l*self.shap_bg_range[0]),l-1,0)]
            hi_bg = temp[clip(int(l*self.shap_bg_range[1]),l-1,0)]
            lo_samp = temp[clip(int(l*self.shap_samp_range[0]),l-1,0)]
            hi_samp = temp[clip(int(l*self.shap_samp_range[1]),l-1,0)]
        else:
            lo_bg = self.shap_bg_range[0]
            hi_bg = self.shap_bg_range[1]
            lo_samp = self.shap_samp_range[0]
            hi_samp = self.shap_samp_range[1]
            
        return lo_bg, hi_bg, lo_samp, hi_samp
    
    def _get_mean_shap_values(self, model, X_bg, X_samp):
        background =  X_bg[np.random.choice(len(X_bg),self.shap_n_background)]
        sample = X_samp[np.random.choice(len(X_samp),self.shap_n_sample)]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=Warning)
            if self.shap_explainer.lower() =='gradient':
                e = shap.GradientExplainer(model, background)
            elif self.shap_explainer.lower() =='deep':
                e = shap.DeepExplainer(model, background)
            shap_values = e.shap_values(sample)[0]
        return shap_values.mean(0)
    
    def __call__(self,model,X_test,X_train):
        if self.include_trainset:
            X_test = np.concatenate([X_test,X_train])
        preds = model.predict(X_test)
        lo_bg, hi_bg, lo_samp, hi_samp = self._get_thresholds(preds)
        preds = preds.ravel()
        X_bg = X_test[(preds>=lo_bg)&(preds<=hi_bg)]
        X_samp = X_test[(preds>=lo_samp)&(preds<=hi_samp)]
        
        shap_values = (self._get_mean_shap_values(model,X_bg,X_samp) + \
                        -self._get_mean_shap_values(model,X_samp,X_bg))/2
        
        return shap_values
    