import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from scipy.stats import norm

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
    # plot survival rate...
    if isinstance(lambda_path,dict):
        lambda_path = lambda_path[list(lambda_path.keys())[0]]
    if isinstance(cv_mean_score,dict):
        cv_mean_score = np.array([np.squeeze(data) for _, data in cv_mean_score.items()])
        cv_mean_score = cv_mean_score.reshape(-1, len(lambda_path))
        cv_mean_score = cv_mean_score.mean(0)
    if isinstance(cv_standard_error,dict):
        cv_standard_error = np.array([np.squeeze(data) for _, data in cv_standard_error.items()])    
        cv_standard_error = cv_standard_error.reshape(-1, len(lambda_path))
        cv_standard_error = cv_standard_error.mean(0)
    
    if isinstance(lambda_val, dict):
        lambda_val = np.array([np.squeeze(data) for _, data in lambda_val.items()])
        
    if isinstance(coef_path,dict):
        coef_path = np.array([np.squeeze(data) for _, data in coef_path.items()])    
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