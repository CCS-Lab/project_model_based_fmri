from scipy.stats import spearmanr, pearsonr, linregress
from sklearn.metrics import mean_squared_error, accuracy_score, roc_curve, auc
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from nilearn import plotting
from mbfmri.utils.config import DEFAULT_MODULATION_SUFFIX, DEFAULT_FEATURE_SUFFIX, \
                    DEFAULT_TIMEMASK_SUFFIX, DEFAULT_SIGNAL_SUFFIX 



def plot_mosaic(img_path,
                save,
               save_path,
               coord_num=7):
    
    img_path = str(img_path)
    plt.figure(figsize=(3*coord_num,9))
    plotting.plot_stat_map(img_path, display_mode='x',cut_coords=coord_num,axes=plt.subplot(3, 1,1))
    plotting.plot_stat_map(img_path, display_mode='y',cut_coords=coord_num,axes=plt.subplot(3, 1,2))
    plotting.plot_stat_map(img_path, display_mode='z',cut_coords=coord_num,axes=plt.subplot(3, 1,3))
    if save:
        plt.savefig(Path(save_path) / f"mosaic_plot_{Path(img_path).name.split('.')[0]}.png",bbox_inches='tight')
    plt.show()

def plot_surface_interactive(img_path,
                             save,
                             save_path):
    img_path = str(img_path)
    if save:
        view = plotting.view_img_on_surf(img_path, 
                                     threshold='90%',
                                     surf_mesh='fsaverage') 
        view.save_as_html(Path(save_path) / f"surface_plot_{Path(img_path).name.split('.')[0]}.html")
        
    
def plot_slice_interactive(img_path,
                           save,
                          save_path):
    img_path = str(img_path)
    if save:
        view = plotting.view_img(img_path) 
        view.save_as_html(Path(save_path) / f"slice_plot_{Path(img_path).name.split('.')[0]}.html")


    
def plot_violinwithscatter(dataframe,
                           score_name,
                           order,
                           title,
                           save,
                           save_path):
    
    plt.figure(figsize=(8, 8))
    
    sns.violinplot(x="type", y=score_name, data=dataframe, order=order)
    sns.stripplot(x="type", y=score_name, data=dataframe, order=order,color='black',alpha=0.5)
    
    lss = [ 'dashed', 'dashdot', 'dotted']
    
    for type_name,ls in zip(order,lss):
        mean = dataframe[dataframe['type']==type_name][score_name].array.mean()
        plt.axhline(mean, ls=ls,label=type_name+"_mean",color='k',alpha=.6)
        
    plt.legend()
    plt.title(title)
    if save:
        dataframe.to_csv(Path(save_path)/'data.csv',sep='\t')
        plt.savefig(Path(save_path)/f'plot_{score_name}.png',bbox_inches='tight')
    plt.show()

def get_scores_dataframe(y_train,
                           y_test,
                           pred_train,
                           pred_test,
                           scorer,
                           score_name,
                           save='False',
                           save_path='.'):
    scores_train = []

    for p,y in zip(pred_train,y_train):
        score = scorer(p.ravel(),y.ravel())
        scores_train.append(score)
        
    scores_test = []
    for p,y in zip(pred_test,y_test):
        score = scorer(p.ravel(),y.ravel())
        scores_test.append(score)
        
    data = pd.DataFrame({score_name: scores_train+scores_test,
                          'type':['train']*len(scores_train)+['test']*len(scores_test)})
    
    if save:
        data.to_csv(Path(save_path)/'data.csv',sep='\t')
        
    return data


def plot_r(y_train,
                  y_test,
                  pred_train,
                  pred_test,
                  save,
                  save_path,
                  pval_threshold=0.05):
    
    
    def get_corrected_r(arrs1,arrs2):
        rs = []
        pvals = []
        for a1,a2 in zip(arrs1,arrs2):
            _,_, r, pv,_ = linregress(a1.ravel(),a2.ravel())
            rs.append(r)
            pvals.append(pv)
        return np.array(rs)[fdrcorrection(pvals, alpha=pval_threshold)[0]]
    
    r_train = get_corrected_r(y_train,pred_train)
    r_test = get_corrected_r(y_test,pred_test)
    
    data = pd.DataFrame({'r': np.concatenate([r_train,r_test]),
                          'type':['train']*len(r_train)+['test']*len(r_test)})
    
    title = f'R. FDR corrected. p<{pval_threshold}'
    
    plot_violinwithscatter(data,
                           'r',
                           ['train','test'],
                           title,
                           save,
                           save_path)
    
def plot_pearsonr(y_train,
                  y_test,
                  pred_train,
                  pred_test,
                  save,
                  save_path,
                  pval_threshold=0.05):
    
    
    def get_corrected_pearsonr(arrs1,arrs2):
        rs = []
        pvals = []
        for a1,a2 in zip(arrs1,arrs2):
            r,pv = pearsonr(a1.ravel(),a2.ravel())
            rs.append(r)
            pvals.append(pv)
        return np.array(rs)[fdrcorrection(pvals, alpha=pval_threshold)[0]]
        
    r_train = get_corrected_pearsonr(y_train,pred_train)
    r_test = get_corrected_pearsonr(y_test,pred_test)
    
    data = pd.DataFrame({'pearsonr': np.concatenate([r_train,r_test]),
                          'type':['train']*len(r_train)+['test']*len(r_test)})
    
    title = f'Pearson R. FDR corrected. p<{pval_threshold}'
    
    plot_violinwithscatter(data,
                           'pearsonr',
                           ['train','test'],
                           title,
                           save,
                           save_path)
def plot_spearmanr(y_train,
                  y_test,
                  pred_train,
                  pred_test,
                  save,
                  save_path,
                  pval_threshold=0.05):
    
    
    def get_corrected_spearmanr(arrs1,arrs2):
        rs = []
        pvals = []
        for a1,a2 in zip(arrs1,arrs2):
            r,pv = spearmanr(a1.ravel(),a2.ravel())
            rs.append(r)
            pvals.append(pv)
        return np.array(rs)[fdrcorrection(pvals, alpha=pval_threshold)[0]] 
    
    r_train = get_corrected_spearmanr(y_train,pred_train)
    r_test = get_corrected_spearmanr(y_test,pred_test)
    
    data = pd.DataFrame({'spearmanr': np.concatenate([r_train,r_test]),
                          'type':['train']*len(r_train)+['test']*len(r_test)})
    
    title = f'Spearman R. FDR corrected. p<{pval_threshold}'
    
    plot_violinwithscatter(data,
                           'spearmanr',
                           ['train','test'],
                           title,
                           save,
                           save_path)

def plot_mse(y_train,
              y_test,
              pred_train,
              pred_test,
              save,
              save_path):
    
    data = get_scores_dataframe(y_train,
                           y_test,
                           pred_train,
                           pred_test,
                           mean_squared_error,
                           'mse')
    
    title = f'Mean Squeared Error (MSE)'
    plot_violinwithscatter(data,
                           'mse',
                           ['train','test'],
                           title,
                           save,
                           save_path)
    
def plot_accuracy(y_train,
              y_test,
              pred_train,
              pred_test,
              save,
              save_path):
    
    pred_train = [(v > .5) *1 for v in pred_train]
    pred_test = [(v > .5) *1 for v in pred_test]
    
    data = get_scores_dataframe(y_train,
                           y_test,
                           pred_train,
                           pred_test,
                           accuracy_score,
                           'accuracy',
                           save,
                           save_path)
    
    
    title = 'Accuracy'
    plot_violinwithscatter(data,
                           'accuracy',
                           ['train','test'],
                           title,
                           save,
                           save_path)

def plot_roc(y_train,
              y_test,
              pred_train,
              pred_test,
              save,
              save_path):
    
    # copied from 
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
    
    y_test= np.concatenate(y_test,0)
    pred_test = np.concatenate(pred_test,0)
    fpr, tpr, _ = roc_curve(y_test.ravel(), pred_test.ravel())
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(8, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    if save:
        plt.savefig(Path(save_path)/f'plot_roccurve.png',bbox_inches='tight')
    plt.show()

def plot_data(mbmvpa_layout, 
              subject,
              run,
              feature_name,
              task_name,
              process_name,
              t_r,
              session=None,
              w=14, 
              h=7, 
              fontsize=15,
              save=False,
              save_path=None,
              show=True):
    
    if not show:
        plt.ioff()
        
    run_kwargs = {'subject':subject,
                 'run':run,
                 'task':task_name}
    
    if session is not None:
        run_kwargs['session'] = session
        
    files = [mbmvpa_layout.get(desc=process_name,suffix=DEFAULT_SIGNAL_SUFFIX,**run_kwargs),
             mbmvpa_layout.get(desc=process_name,suffix=DEFAULT_MODULATION_SUFFIX,**run_kwargs),
             mbmvpa_layout.get(desc=process_name,suffix=DEFAULT_TIMEMASK_SUFFIX,**run_kwargs),
             mbmvpa_layout.get(desc=feature_name,suffix=DEFAULT_FEATURE_SUFFIX,**run_kwargs)]
    
    # sanity check
    for f in files:
        if len(f) != 1:
            return -1

    signal_file, modulation_file,\
        timemask_file, feature_file = [f[0] for f in files]
    
    fig = plt.figure(figsize=(w,h*2))
    add_voxel_feature_subplot(feature_file, t_r, ax_idx=1, fig=fig, total_number=2,fontsize=fontsize, skip_xlabel=True)
    add_latent_process_subplot(modulation_file, signal_file, timemask_file, t_r, ax_idx=2, fig=fig, total_number=2,fontsize=fontsize)
    
    if save:
        if session is not None:
            file_name = f'sub-{subject}_ses-{session}_task-{task_name}_run-{run}_plot.png'
        else:
            file_name = f'sub-{subject}_task-{task_name}_run-{run}_plot.png'
        plt.savefig(Path(save_path)/file_name,bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.ion()
    plt.close()
    return 1
    
def add_voxel_feature_subplot(feature_file, 
                              t_r, 
                              ax_idx=1, 
                              fig=None, 
                              total_number=1,
                              fontsize=15,
                              skip_xlabel=False):
    
    if fig is None:
        fig = plt.figure(figsize=(14,7))
        
    feature_file = Path(feature_file)
    ax = fig.add_subplot(total_number, 1, ax_idx)
    feature = np.load(feature_file)
    ax.imshow(feature.T, interpolation='nearest', aspect='auto', cmap='gray')
    if not skip_xlabel:
        ax.set_xlabel('time (s)',fontsize=fontsize)
    ax.set_ylabel('voxel',fontsize=fontsize)
    n_scan = feature.shape[0]
    xticks = np.arange(0,n_scan+1,n_scan//10)
    xticklabels = xticks*t_r
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_title(Path(feature_file).stem, fontsize=fontsize)
    
    
def add_latent_process_subplot(modulation_file, 
                               signal_file, 
                               timemask_file, 
                               t_r, 
                               ax_idx=1, 
                               fig=None, 
                               total_number=1,
                               fontsize=15,
                               skip_xlabel=False):
    if fig is None:
        fig = plt.figure(figsize=(14,7))
    
    
    modulation_file = Path(modulation_file)
    signal_file = Path(signal_file)
    timemask_file = Path(timemask_file)
    signal = np.load(signal_file)
    modulation = pd.read_table(modulation_file)
    
    n_scan = len(signal)
    mod_array = np.zeros(n_scan)

    for _, row in modulation.iterrows():
        onset_idx = int(float(row['onset'])/t_r)
        end_idx = int((float(row['onset'])+float(row['duration']))/t_r)
        mod_val = float(row['modulation'])
        mod_array[onset_idx:end_idx+1]=mod_val
                
        
    timemask = np.load(timemask_file)
    timemask_ranges = []
    is_valid = -1
    for i,v in enumerate(1-timemask):
        if v > 0 and is_valid <=0:
            sid = i
            is_valid = 1
        elif v<= 0 and is_valid >0:
            timemask_ranges.append((sid,i))
            is_valid = 0
        else:
            continue
    if is_valid >0:
        timemask_ranges.append((sid,len(timemask)))
    
    xticks = np.arange(0,n_scan+1,n_scan//10)
    xticklabels = xticks*t_r
    
    
    ax_mod = fig.add_subplot(total_number*2, 1, 2*ax_idx-1)
    ax_mod.stem(mod_array, label='modulation',linefmt='black', markerfmt=' ',basefmt="black")
    if len(timemask_ranges) > 0:
        ax_mod.axvspan(timemask_ranges[0][0], timemask_ranges[0][1],
                          color='gray', alpha=.3, lw=1, label='masked-out')
        for (si,ei) in timemask_ranges:
            ax_mod.axvspan(si, ei,color='gray', alpha=.3, lw=1)
    ax_mod.set_title(modulation_file.stem, fontsize=fontsize)
    ax_mod.get_xaxis().set_visible(False)
    ax_mod.set_ylabel('value (a.u.)',fontsize=fontsize)
    ax_mod.set_xlim([xticks[0],xticks[-1]])
    ax_mod.legend(loc='upper right')
    
    
    ax_signal = fig.add_subplot(total_number*2, 1, 2*ax_idx)
    ax_signal.plot(signal, label='signal', color='red')
    if len(timemask_ranges) > 0:
        ax_signal.axvspan(timemask_ranges[0][0], timemask_ranges[0][1],
                          color='gray', alpha=.3, lw=1, label='masked-out')
        for (si,ei) in timemask_ranges[1:]:
            ax_signal.axvspan(si, ei,color='gray', alpha=.3, lw=1)
    ax_signal.set_title(signal_file.stem, fontsize=fontsize)
    ax_signal.set_ylabel('value (a.u.)',fontsize=fontsize)
    if not skip_xlabel:
        ax_signal.set_xlabel('time (s)',fontsize=fontsize)
    ax_signal.set_xticks(xticks)
    ax_signal.set_xticklabels(xticklabels)
    ax_signal.set_xlim([xticks[0],xticks[-1]])
    ax_signal.legend(loc='upper right')
    
    