import matplotlib.pyplot as plt
from scipy import stats
import numpy as np


def plot_sham_result(errors, sham_errors, save_path=None):
    t, p = stats.ttest_ind(errors, sham_errors)
    text = ("ns   p > 0.05\n"+
            "*    p ≤ 0.05\n"+
            "**   p ≤ 0.01\n"+
            "***  p ≤ 0.001")

    if p > 0.05:
        significance_mark = 'ns'
    elif p > 0.01:
        significance_mark = "*"
    elif p > 0.001:
        significance_mark = "**"
    else:
        significance_mark = "***"

    plt.figure(figsize=(5, 8))
    plt.title('Error compared with sham training', fontsize=20)
    plt.boxplot([errors, sham_errors], labels=['original','sham'], widths=0.6)
    yticks = plt.yticks()[0]
    x1, x2 = plt.xticks()[0]
    ytick_gap = yticks[-1]-yticks[-2]
    ytick_max = yticks[-1]
    y = ytick_max+ytick_gap
    h = ytick_gap/2
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, color='black')
    plt.text((x1+x2)/2,y+h,significance_mark,fontsize=20)
    plt.ylabel('error (a.u.)', fontsize=20)
    plt.text(2.6,yticks[0]+2*h,text,fontsize=12)
    plt.xticks(fontsize=20)
    
    if save_path is not None:
        plt.savefig(save_path/'sham_result.png',bbox_inches='tight')
            
    plt.show()
    