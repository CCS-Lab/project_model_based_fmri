#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21



import numpy as np
import shap
import warnings
from tensorflow.compat.v1.keras.backend import get_session
import tensorflow as tf
from scipy.stats import linregress
from statsmodels.stats.multitest import fdrcorrection
tf.compat.v1.disable_v2_behavior()
#shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
# https://github.com/slundberg/shap/tree/9411b68e8057a6c6f3621765b89b24d82bee13d4

class Explainer():
    
    r"""
    Explainer interprets Tensorflow Keras models based on SHAP by Lundberg et al., 2017.
    This module is aiming to make voxel-level interpretation of MVPA models.

    By SHAP, Feature attribution of each voxel is calculated for sample from input distribution.
    
    The weights are defined as a linear slope of activity-attribution for each voxel, 
    which corresponds with reading coefficients in one-layer linear models. 

    In general such activityy-attribution relation ship is similar 
    to each kernel in kernel regression. Here, for intuitivity and simplicity, only 
    linear kernel is considered.

    The sampled SHAP values can be saved as well, allowing the further analysis on 
    how voxel activities modulate the corresponding voxel attribution. 

    
    Parameters
    ----------

    shap_explainer : str, default="deep"
        Name for SHAP algorithm to interprete Keras models.
        Two algorithms, "deep" and "gradient," are in options.
        Please refer to the package repo (https://github.com/slundberg/shap/),

    shap_n_background : int, default=100
        Number of background to be used as reference in SHAP calculation.
        The number of sample will be retrieved from the given input data.

    shap_n_sample : int, default=100
        Number of sample to be used as input to explain by SHAP.
        The number of sample will be retrieved from the given input data.
        The SHAP values for each feature and each sample will be obtained.

    include_trainset : bool, default=True
        Iindicate if including train set in the input pool 
        from which the backgrounds and sample will be retrieved.
    
    pval_threshold : float, default=0.05
        Threshold value for false detection rate (FDR) correction. 
        FDR will be done for threshold out insignificantly 
        estimated slopes in activity-attribution.

    voxel_mask : nibabel.Nifti1Image, default=None
        Nii object for mask inputs. It is used for flattening input data 
        if the input is provided in 3D shape (when CNN is used).
    
    
    """

    def __init__(self,
                 shap_explainer='deep',
                 shap_n_background = 100,
                 shap_n_sample = 100,
                 include_trainset=True,
                 pval_threshold=0.05,
                 voxel_mask=None):
    
        self.shap_explainer = shap_explainer
        self.shap_n_background = shap_n_background
        self.shap_n_sample = shap_n_sample
        self.include_trainset = include_trainset
        self.pval_threshold = pval_threshold
        self.voxel_mask = voxel_mask
        

    def __call__(self,model,X_test,X_train):
        if self.include_trainset:
            X_test = np.concatenate([X_test,X_train])
        preds = model.predict(X_test)
        preds = preds.ravel()
        X_bg = X_test
        X_samp = X_test
        background =  X_bg[np.random.choice(len(X_bg),self.shap_n_background)]
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
            
        
        # masking & flattening
        if len(sample.shape )>= 4:
            mask = self.voxel_mask.get_fdata() > 0
            sample = sample.transpose(1,2,3,0)[mask].transpose(1,0)
            shap_values = shap_values.transpose(1,2,3,0)[mask].transpose(1,0)
        
        # get slopes
        regs = [linregress(sample[:,i],shap_values[:,i]) for i in range(sample.shape[1])]
        pvalue = [reg.pvalue for reg in regs]
        valid,_ = fdrcorrection(pvalue, alpha=self.pval_threshold)
        slopes = np.array([reg.slope for reg in regs])
        slopes[~valid] = 0
        
        output = {}
        output['weights'] = slopes
        output['shap_values']=shap_values
        output['shap_sample']=sample
        output['shap_pred'] = preds[sample_idxs]
            
        return output
        