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


from mbmvpa.core.engine import run_mbmvpa
from pathlib import Path

bids_layout = "tests/test_example"
report_path = "tests/test_report"
Path(report_path).mkdir(exist_ok=True)

def test_adjust(row):
    ## rename data in a row to the name which can match hbayesdm.ra_prospect requirements ##
    row["gamble"] = 1 if row["respcat"] == 1 else 0
    row["cert"] = 0
    return row

_ = run_mbmvpa(bids_layout=bids_layout,
               dm_model='ra_prospect',
               mvpa_model='cnn',
               feature_name='zoom2',
               task_name='mixedgamblestask',
               process_name='SUgamble',
               #process_name='gain',
               skip_compmodel=True,
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
               pval_threshold=5)

print("TEST PASS!")