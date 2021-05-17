#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.03.23
    
from glmnet import ElasticNet
import numpy as np
from pathlib import Path
from mbmvpa.models.mvpa_general import MVPA_Base, MVPA_CV
from mbmvpa.utils.report import build_elasticnet_report_functions


class MVPACV_ElasticNet(MVPA_CV):
    
    r"""
    
    **MVPACV_ElasticNet** is for providing cross-validation (CV) framework with ElasticNet as an MVPA model.
    Users can choose the option for CV (e.g. 5-fold or leave-one-subject-out), and the model specification.
    Also, users can modulate the configuration for reporting function which includes making brain map (nii), 
    and plots.
    
    Parameters
    ----------
    
    X_dict : dict{str : numpy.ndarray}
        A dictionary for the input voxel feature data which can be indexed by subject IDs.
        Each voxel feature array should be in shape of [time len, voxel feature name]
    y_dict : dict{str : numpy.ndarray}
        A dictionary for the input latent process signals which can be indexed by subject IDs.
        Each signal should be in sahpe of [time len, ]
    voxel_mask : nibabel.nifti1.Nifti1Image
        A brain mask image (nii) used for masking the fMRI images. It will be used to reconstruct a 3D image
        from flattened array of model weights.
    method : str, default='5-fold'
        The name for type of cross-validation to use. 
        Currently, two options are available.
            - "N-fold" : *N*-fold cross-valiidation
            - "N-lnso" : leave-*N*-subjects-out
            
        If the "N" should be a positive integer and it will be parsed from the input string. 
        In the case of lnso, N should be >= 1 and <= total subject # -1.
    n_cv_repeat : int, default=1
        The number of repetition of the entire cross-validation.
        Larger the number, (normally) more stable results and more time required.
    cv_save : bool, default=True
        indictates save results or not
    cv_save_path : str or pathlib.PosixPath, default="."
        A path for saving results
    experiment_name : str, default="unnamed"
        A name for a single run of this analysis
        It will be included in the name of the report folder created.
    alpha : float, default=0.001
        A value between 0 and 1, indicating the mixing parameter in ElasticNet.
        *penalty* = [alpha * L1 + (1-alpha)/2 * L2] * lambda
    n_sample : int, default=30000
        Max number of samples used in a single fitting.
        If the number of data is bigger than *n_sample*, sampling will be done for 
        each model fitting.
        This is for preventing memory overload.
    max_lambda : float, default=10
        The maximum value of lambda in lambda search space.
        The lambda search space is used when searching the best lambda value.
    min_lambda_ratio : float, default=1e-4
        The ratio of minimum lambda value to maximum value. 
        With this ratio, a log-linearly scaled lambda space will be created.
    lambda_search_num : int, default=100
        The number of points in lambda search space. 
        Bigger the number, finer will the lambda searching be.
    n_jobs : int, default=16
        The number of cores used in fitting ElasticNet
    n_splits : int, default=5
        The number of fold used in inner cross-validation,
        which aims to find the best lambda value.
    confidence_interval : float, default=.99
        Confidence level for plotting CV errors in lambda searching.
    n_coef_plot : int, default=150
        The number of samples for plotting coefficient values in lambda searching.
    map_type : str, default="z"
        The type of making brain map. 
            - "z" : z-map will be created using all the weights from CV experiment.
            - "t" : t-map will be created using all the weights from CV experiment.v
    sigma : float, default=1
        The sigma value for running Gaussian smoothing on each of reconstructed maps, 
        before integrating maps to z- or t-map.
    
    """
    
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
                 n_sample=30000,
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
                                    n_sample=n_sample,
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
        A value between 0 and 1, indicating the mixing parameter in ElasticNet.
        *penalty* = [alpha * L1 + (1-alpha)/2 * L2] * lambda
    n_sample : int, default=30000
        Max number of samples used in a single fitting.
        If the number of data is bigger than *n_sample*, sampling will be done for 
        each model fitting.
        This is for preventing memory overload.
    max_lambda : float, default=10
        The maximum value of lambda in lambda search space.
        The lambda search space is used when searching the best lambda value.
    min_lambda_ratio : float, default=1e-4
        The ratio of minimum lambda value to maximum value. 
        With this ratio, a log-linearly scaled lambda space will be created.
    lambda_search_num : int, default=100
        The number of points in lambda search space. 
        Bigger the number, finer will the lambda searching be.
    n_jobs : int, default=16
        The number of cores used in fitting ElasticNet
    n_splits : int, default=5
        The number of fold used in inner cross-validation,
        which aims to find the best lambda value.
    
    """
    
    
    def __init__(self,
                 alpha=0.001,
                 n_sample=30000,
                 max_lambda=10,
                 min_lambda_ratio=1e-4,
                 lambda_search_num=100,
                 n_jobs=16,
                 n_splits=5,
                 **kwargs):
        
        # penalty = [alpha * L1 + (1-alpha)/2 * L2] * lambda
        
        self.n_jobs = n_jobs
        self.n_splits = n_splits
        self.n_sample = n_sample
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
