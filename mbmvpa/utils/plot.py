from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pdb


def plot_pearsonr(y_train,
                  y_test,
                  pred_train,
                  pred_test,
                  save,
                  save_path,
                  pval_threshold=None):
    
    
    r_train = []
    for y,p in zip(y_train,pred_train):
        r,pval = pearsonr(y.ravel(),p.ravel())
        if pval_threshold is not None and pval < pval_threshold:
            r_train.append(r)
            
    r_test = []
    for y,p in zip(y_test,pred_test):
        r,pval = pearsonr(y.ravel(),p.ravel())
        if pval_threshold is not None and pval < pval_threshold:
            r_test.append(r)
            
    
    plt.figure(figsize=(4, 8))
    plt.boxplot([r_train, r_test], labels=['train','test'], widths=0.6)
    if save:
        plt.savefig(save_path/f'plot_pearsonr.png',bbox_inches='tight')
    plt.show()