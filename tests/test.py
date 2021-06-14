#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.05.03

"""
This is a test code for testing the package with
a simplest input. The test file is made from Tom et al., 2007.

https://openneuro.org/datasets/ds000005/versions/00001 

"""

from mbfmri.core.engine import run_mbfmri, run_mbmvpa
from pathlib import Path

## GENERAL SETTING
bids_layout = "tests/test_example"
report_path = "tests/test_report"
Path(report_path).mkdir(exist_ok=True)

def test_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.ra_prospect requirements ##
    row["gamble"] = 1 if row["respcat"] == 1 else 0
    row["cert"] = 0
    return row
'''
## TEST MVPA BASIC

_ = run_mbfmri(bids_layout=bids_layout,
               dm_model='ra_prospect',
               mvpa_model='elasticnet',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='SUgamble',
               skip_compmodel=False,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               overwrite=True,
               gpu_visible_devices = [2],
               n_batch=4,
               pval_threshold=5,
              refit_compmodel=True)

'''
## TEST MODELCOMPARISON
_ = run_mbmvpa(bids_layout=bids_layout,
               dm_model=['ra_prospect','ra_noRA','ra_noLA'],
               mvpa_model='elasticnet',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='SUgamble',
               skip_compmodel=False,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               reports=['pearsonr','r','mse','spearmanr'],
               n_batch=4,
               pval_threshold=5)
'''
## TEST PRECALCULATED PROCESS
_ = run_mbfmri(bids_layout=bids_layout,
               mvpa_model='elasticnet',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='gain',
               skip_compmodel=True,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               n_batch=4,
               pval_threshold=5)

## TEST MVPA-MLP
_ = run_mbfmri(bids_layout=bids_layout,
               mvpa_model='mlp',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='gain',
               skip_compmodel=True,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               n_batch=4,
               pval_threshold=5)

## TEST MVPA-CNN
_ = run_mbfmri(bids_layout=bids_layout,
               mvpa_model='cnn',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='gain',
               skip_compmodel=True,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               n_batch=4,
               pval_threshold=5)

## TEST MVPA-HIERARCHICAL
_ = run_mbfmri(analysis='mbmvpah',
               bids_layout=bids_layout,
               dm_model='ra_prospect',
               mvpa_model='elasticnet',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='SUgamble',
               skip_compmodel=False,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               n_batch=4,
               pval_threshold=5)

## TEST GLM
_ = run_mbfmri(analysis='glm',
              report_path=report_path,
              bids_layout=bids_layout,
              dm_model='ra_prospect',
              task_name='mixedgamblestask',
              process_name='SUgamble',
              adjust_function=test_adjust,
              overwrite=False,
              overwrite_latent_process=True,
              refit_compmodel=False)

## TEST MVPA-MLP Logistic
_ = run_mbfmri(bids_layout=bids_layout,
               logistic=True,
               mvpa_model='mlp',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='gain',
               skip_compmodel=True,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               n_batch=4,
               pval_threshold=5)
'''
## TEST ElasticNet Logistic
_ = run_mbfmri(bids_layout=bids_layout,
               logistic=True,
               mvpa_model='elasticnet',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='gain',
               skip_compmodel=True,
               report_path=report_path,
               adjust_function=test_adjust,
               n_core=4,
               nchain=2,
               nwarmup=50,
               niter=200,
               n_thread=4,
               method='5-fold',
               gpu_visible_devices = [2],
               n_batch=4,
               reports=['brainmap','accuracy','roc'],
               pval_threshold=5)
print("TEST PASS!")