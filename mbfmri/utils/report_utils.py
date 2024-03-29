from mbfmri.utils.coef2map import get_map
from mbfmri.utils.plot import *
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import spearmanr, pearsonr, linregress
from sklearn.metrics import mean_squared_error, roc_auc_score, accuracy_score
import re

class Metric_AUC():
    def __init__(self):
        return
    
    def __call__(self,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        output = {'train' : roc_auc_score(y_train,pred_train),
                 'test' : roc_auc_score(y_test,pred_test)}
        return output
    
class Metric_Accuracy():
    def __init__(self):
        return
    
    def __call__(self,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        output = {'train' : accuracy_score(y_train,(pred_train > .5) *1),
                 'test' : accuracy_score(y_test,(pred_test > .5) *1)}
        return output
    
class Metric_MSE():
    def __init__(self):
        return
    
    def __call__(self,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        output = {'train' : mean_squared_error(y_train,pred_train),
                 'test' : mean_squared_error(y_test,pred_test)}
        return output
    

    
class Metric_PearsonR():
    def __init__(self):
        return
    
    def __call__(self,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        r_train, pv_train = pearsonr(y_train.ravel(),pred_train.ravel())
        r_test, pv_test = pearsonr(y_test.ravel(),pred_test.ravel())
        output = {'train' : r_train,
                  'pvalue_train': pv_train,
                  'test' : r_test,
                  'pvalue_test' : pv_test}
        return output
    
    
class Metric_SpearmanR():
    def __init__(self):
        return
    
    def __call__(self,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        r_train, pv_train = spearmanr(y_train.ravel(),pred_train.ravel())
        r_test, pv_test = spearmanr(y_test.ravel(),pred_test.ravel())
        output = {'train' : r_train,
                  'pvalue_train': pv_train,
                  'test' : r_test,
                  'pvalue_test' : pv_test}
        return output
    
class Metric_R():
    def __init__(self):
        return
    
    def __call__(self,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        _,_, r_train, pv_train,_ = linregress(y_train.ravel(),pred_train.ravel())
        _,_, r_test, pv_test,_ = linregress(y_test.ravel(),pred_test.ravel())
        output = {'train' : r_train,
                  'pvalue_train': pv_train,
                  'test' : r_test,
                  'pvalue_test' : pv_test}
        return output

class Plot_ROC():
    
    def __init__(self):
        return
    
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        plot_roc(save=save,
                 save_path=save_path,
                y_train=y_train,
                y_test=y_test,
                pred_train=pred_train,
                pred_test=pred_test)
        
class Plot_Accuracy():
    
    def __init__(self):
        return
    
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        plot_accuracy(save=save,
                 save_path=save_path,
                y_train=y_train,
                y_test=y_test,
                pred_train=pred_train,
                pred_test=pred_test)
        
class Plot_MSE():
    
    def __init__(self):
        return
    
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        plot_mse(save=save,
                 save_path=save_path,
                y_train=y_train,
                y_test=y_test,
                pred_train=pred_train,
                pred_test=pred_test)


class Report_BrainMap():
    
    def __init__(self,
                voxel_mask,
                experiment_name='unnamed',
                standardize=True,
                map_smoothing_fwhm=6,
                map_threshold=0,
                cluster_threshold=0):
        
        self.voxel_mask = voxel_mask
        self.experiment_name = experiment_name
        self.standardize = standardize
        self.smoothing_fwhm = map_smoothing_fwhm
        self.threshold = map_threshold
        self.cluster_threshold = cluster_threshold
        
    def __call__(self,
                 save,
                 save_path,
                 weights
                ):
        
        
        if isinstance(weights, dict):
            coefs = np.array([np.squeeze(data) for _, data in weights.items()]) 
        else:
            coefs = weights 
            
        nii, img_path = get_map(coefs, self.voxel_mask, self.experiment_name,
                standardize=self.standardize, save_path=save_path, smoothing_fwhm=self.smoothing_fwhm,
                               threshold=self.threshold,cluster_threshold=self.cluster_threshold)
        
        plot_mosaic(img_path,save,save_path)
        plot_surface_interactive(img_path,save,save_path)
        plot_slice_interactive(img_path,save,save_path)
        return nii
    
    
def _smooth(preds,shap_values,n_bin=10,sigma_scale=.3):
        xub = preds.std()*1.96 + preds.mean()
        prm = preds.mean()
        prs = preds.std()
        sigma = prs * sigma_scale
        x = np.linspace(prm-1.96*prs,prm+1.96*prs,n_bin)

        def gk(x,m,s):
            return 1/s/(2*np.pi)**.5*np.exp(-(x-m)**2/2/s**2)


        shap_values_smthed = None

        for v in x:
            weights= gk(preds,v,sigma)
            smthed = weights.dot(shap_values)/weights.sum()
            smthed = np.expand_dims(smthed,0)
            if shap_values_smthed is None:
                shap_values_smthed=smthed
            else:
                shap_values_smthed  = np.vstack((shap_values_smthed,smthed))

        return shap_values_smthed
    

class Plot_ElasticNet():
    
    def __init__(self,
                confidence_interval=.99,
                 n_coef_plot=150):
        
        self.confidence_interval = confidence_interval
        self.n_coef_plot = n_coef_plot
    
    def __call__(self,
                 save,
                 save_path,
                 cv_mean_score,
                 cv_standard_error,
                 lambda_path,
                 lambda_best,
                 coef_path):
        
        
        plot_elasticnet_result(save_root=save_path, 
                           save=save,
                           cv_mean_score=cv_mean_score, 
                           cv_standard_error=cv_standard_error,
                           lambda_path=lambda_path,
                           lambda_val=lambda_best,
                           coef_path=coef_path,
                           confidence_interval=self.confidence_interval,
                           n_coef_plot=self.n_coef_plot)
        
class Plot_PearsonR():
    
    def __init__(self, pval_threshold=0.01):
        self.pval_threshold = pval_threshold
        
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        plot_pearsonr(save=save,
                     save_path=save_path,
                        y_train=y_train,
                        y_test=y_test,
                        pred_train=pred_train,
                        pred_test=pred_test,
                     pval_threshold=self.pval_threshold)

class Plot_R():
    
    def __init__(self, pval_threshold=0.01):
        self.pval_threshold = pval_threshold
        
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        plot_r(save=save,
                     save_path=save_path,
                        y_train=y_train,
                        y_test=y_test,
                        pred_train=pred_train,
                        pred_test=pred_test,
                     pval_threshold=self.pval_threshold)
        
class Plot_SpearmanR():
    
    def __init__(self, pval_threshold=0.01):
        self.pval_threshold = pval_threshold
        
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        plot_spearmanr(save=save,
                     save_path=save_path,
                        y_train=y_train,
                        y_test=y_test,
                        pred_train=pred_train,
                        pred_test=pred_test,
                     pval_threshold=self.pval_threshold)
        
        