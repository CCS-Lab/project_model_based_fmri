from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import pdb

def plot_pearsonr(y_train,
                  y_test,
                  pred_train,
                  pred_test,
                  save,
                  save_path,
                  pval_threshold=0.05):
    
    
    rs = []
    pvals = []

    for p,y in zip(pred_train,y_train):
        r,pv = pearsonr(p.ravel(),y.ravel())
        rs.append(r)
        pvals.append(pv)
    
    r_train = [r for r, v in zip(rs, fdrcorrection(pvals, alpha=pval_threshold)[0]) if v]
            
    rs = []
    pvals = []
    for p,y in zip(pred_test,y_test):
        r,pv = pearsonr(p.ravel(),y.ravel())
        rs.append(r)
        pvals.append(pv)
    
    r_test = [r for r, v in zip(rs, fdrcorrection(pvals, alpha=pval_threshold)[0]) if v]
    
    plt.figure(figsize=(4, 8))
    plt.boxplot([r_train, r_test], labels=['train','test'], widths=0.6)
    plt.title(f'Pearson R. FDR corrected. p<{pval_threshold}')
    if save:
        plt.savefig(save_path/f'plot_pearsonr.png',bbox_inches='tight')
    plt.show()
    
    
def plot_carpet(voxelfeature):
    
    if not sinstance(voxelfeature, type(np.array([]))):
        voxelfeature=np.load(str(voxelfeature))
    
    plt.figure(figsize=(14,7))
    plt.imshow(voxelfeature.T, interpolation='nearest', aspect='auto', cmap='gray')
    plt.show()