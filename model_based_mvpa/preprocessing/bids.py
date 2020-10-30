import os
import numpy as np
from pathlib import Path

import bids
from bids import BIDSLayout, BIDSValidator

from concurrent.futures import ProcessPoolExecutor
import fMRI
import logging


bids.config.set_option('extension_initial_dot', True)
logging.basicConfig(encoding='utf-8', level=logging.INFO)


def bids_preprocess(root,
                    save_path=None,
                    save=True,
                    single_file=False,
                    zoom=(1, 1, 1),
                    smoothing_fwhm=6,
                    interpolation_func=np.mean,
                    motion_confounds=['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z'],
                    p_value=0.05,
                    task_name='task-zero',
                    standardize=True,
                    ncore=os.cpu_count()):

    logging.info('loading bids dataset.. 0/4')
    layout = BIDSLayout(root, derivatives=True)
    nii_layout = layout.derivatives['fMRIPrep'].get(return_type='file', suffix='bold', extension='nii.gz')
    reg_layout = layout.derivatives['fMRIPrep'].get(return_type='file', suffix='regressors', extension='tsv')
    
    n_subject = len(layout.get_subjects())
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())
    logging.info('done 1/4..')
    
    logging.info('make custom masking.. 1/4')
    root = Path(root)
    mask_path = Path(layout.derivatives['fMRIPrep'].root) / 'mask'

    masked_data, masker, m_true = custom_masking(
        mask_path, p_value, zoom,
        smoothing_fwhm, interpolation_func, standardize
    )
    print(masked_data.shape)
    print(m_true.shape)
    logging.info('done..! 2/4')
    
    logging.info('image preprocessing using %d cores.. 2/4' % ncore)
    params = [[z[0], z[1], motion_confounds, masker, masked_data, i]
            for i, z in enumerate(zip(nii_layout, reg_layout))]

    with ProcessPoolExecutor() as executor:
        X = np.array(list(executor.map(image_preprocess, params)))
        
        if n_session != 0:
            X = X.reshape(n_subject, n_session, n_run, -1, m_true.shape[0])
        else:
            X = X.reshape(n_subject, n_run, -1, m_true.shape[0])
    logging.info('done..! 3/4')

    logging.info('file saving..! 3/4')
    if save:
        if save_path is None:
            sp = Path(layout.derivatives['fMRIPrep'].root) / 'data'
        else:
            sp = Path(save_path)
            
        if not sp.exists():
            sp.mkdir()
        
        if single_file:
            np.save(sp / 'X.npy', X)
        else:
            for i in range(X.shape[0]):
                np.save(sp / f'X_{i+1}.npy', X[i])
        nib.save(masked_data, sp / 'masked_data.nii.gz')
    logging.info('done..! 4/4')
    
    return X, masked_data