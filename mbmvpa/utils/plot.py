from scipy.stats import pearsonr
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import pdb
from pathliib import Path

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
    
    
def plot_data(mbmvpa_layout, 
              subject,
              run,
              feature_name,
              task_name,
              process_name,
              t_r,
              w=14, 
              h=7, 
              fontsize=15):
    
    signal_file = mbmvpa_layout.get(subject=subject, run=run, desc=process_name,suffix='signal')[0]
    modulation_file = mbmvpa_layout.get(subject=subject, run=run, desc=process_name,suffix='modulation')[0]
    timemask_file = mbmvpa_layout.get(subject=subject, run=run, desc=process_name,suffix='timemask')[0]
    feature_file = mbmvpa_layout.get(subject=subject, run=run, desc=feature_name,suffix='voxelfeature')[0]
    
    fig = plt.figure(figsize=(w,h*2))
    add_voxel_feature_subplot(feature_file, t_r, ax_idx=1, fig=fig, total_number=2,fontsize=fontsize, skip_xlabel=True)
    add_latent_process_subplot(modulation_file, signal_file, timemask_file, t_r, ax_idx=2, fig=fig, total_number=2,fontsize=fontsize)
    
    
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
            
    
    
    ax_mod = fig.add_subplot(total_number*2, 1, 2*ax_idx-1)
    ax_mod.stem(mod_array, label='modulation',linefmt='black', markerfmt=' ',basefmt="black")
    for (si,ei) in timemask2unvalidrange(timemask):
        ax_mod.axvspan(si, ei,color='gray', alpha=.3, lw=1, label='masked-out')
    ax_mod.set_title(modulation_file.stem, fontsize=fontsize)
    ax_mod.get_xaxis().set_visible(False)
    ax_mod.set_ylabel('value (a.u.)',fontsize=fontsize)
    ax_mod.legend()
    
    
    ax_signal = fig.add_subplot(total_number*2, 1, 2*ax_idx)
    ax_signal.plot(signal, label='signal', color='red')
    for (si,ei) in timemask2unvalidrange(timemask):
        ax_signal.axvspan(si, ei,color='gray', alpha=.3, lw=1, label='masked-out')
    ax_signal.set_title(signal_file.stem, fontsize=fontsize)
    ax_signal.set_ylabel('value (a.u.)',fontsize=fontsize)
    if not skip_xlabel:
        ax_signal.set_xlabel('time (s)',fontsize=fontsize)
    xticks = np.arange(0,n_scan+1,n_scan//10)
    xticklabels = xticks*t_r
    ax_signal.set_xticks(xticks)
    ax_signal.set_xticklabels(xticklabels)
    ax_signal.legend()
    
    