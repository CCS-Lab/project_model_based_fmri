from .coef2map import get_map
from .plot import plot_pearsonr
from scipy.stats import pearsonr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
import re


class Report_BrainMap():
    
    def __init__(self,
                voxel_mask,
                 experiment_name='unnamed',
                 map_type='z',
                 sigma=1):
        
        self.voxel_mask = voxel_mask
        self.experiment_name = experiment_name
        self.map_type = map_type
        self.sigma = sigma
        
    def __call__(self,
                 save,
                 save_path,
                 weights
                ):
        
        if isinstance(weights, dict):
            coefs = np.array([np.squeeze(data) for _, data in weights.items()]) 
        else:
            coefs = weights.reshape(-1, weights.shape[-1])
            
        get_map(coefs, self.voxel_mask, self.experiment_name,
                map_type=self.map_type, save_path=save_path, sigma=self.sigma)

class Report_ElasticNet():
    
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
        
class Report_PearsonR():
    
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


def plot_elasticnet_result(save_root, 
                           save,
                           cv_mean_score, 
                           cv_standard_error,
                           lambda_path,
                           lambda_val,
                           coef_path,
                           confidence_interval=.99,
                           n_coef_plot=150):
    
    if save:
        save_root = Path(save_root) /'plot'
        save_root.mkdir(exist_ok = True)
    
    # make dictionary as reportable array.
    
    lambda_path = lambda_path[0]
    cv_mean_score = cv_mean_score.reshape(-1, len(lambda_path))
    cv_mean_score = cv_mean_score.mean(0)
    cv_standard_error = cv_standard_error.reshape(-1, len(lambda_path))
    cv_standard_error = cv_standard_error.mean(0)
    coef_path = coef_path.reshape(-1, coef_path.shape[-2], coef_path.shape[-1])
    coef_path = coef_path.mean(0)
    
    plt.figure(figsize=(10, 8))
    plt.errorbar(np.log(lambda_path), cv_mean_score,
                 yerr=cv_standard_error* norm.ppf(1-(1-confidence_interval)/2), 
                 color='k', alpha=.5, elinewidth=1, capsize=2)
    # plot confidence interval
    plt.plot(np.log(lambda_path), cv_mean_score, color='k', alpha=0.9)
    plt.axvspan(np.log(lambda_val.min()), np.log(lambda_val.max()),
                color='skyblue', alpha=.75, lw=1)
    plt.xlabel('log(lambda)', fontsize=20)
    plt.ylabel('cv average MSE', fontsize=20)
    if save:
        plt.savefig(save_root/'plot1.png',bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(10, 8))
    plt.plot(np.log(lambda_path), coef_path[
             np.random.choice(np.arange(coef_path.shape[0]), n_coef_plot), :].T)
    plt.axvspan(np.log(lambda_val.min()), np.log(lambda_val.max()),
                color='skyblue', alpha=.75, lw=1)
    plt.xlabel('log(lambda)', fontsize=20)
    plt.ylabel('coefficients', fontsize=20)
    if save:
        plt.savefig(save_root/'plot2.png',bbox_inches='tight')
    plt.show()
    