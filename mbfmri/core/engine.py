#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.03

from mbfmri.core.glm import run_mbglm, GLM
from mbfmri.core.mvpa import run_mbmvpa, MBMVPA

GLM = GLM
MBMVPA = MBMVPA
run_mbglm = run_mbglm
run_mbmvpa = run_mbmvpa

valid_analysis = ['mvpa','glm']

def run_mbfmri(analysis='mvpa',
              **kwargs):
    
    """Top wrapper function for model-based fMRI analysis.
    It offers GLM approach and MVPA approach.
    
    Parameters
    ----------
    
    analysis: str, default="mvpa"
        Name of approach. "mvpa" will conduct MVPA approach by running *run_mbmvpa*,
        and "glm" will conduct GLM approach by running *run_mbglm*.
    
    kwargs: dict
        Kewarded dictionary for configuration. 
        Please check the documentation of submodules.
        
        `Generating multi-voxel signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.bold>`_
        
        `Generating latent process signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.events>`_
        
        `Data loading document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.data.html#mbfmri.data.loader.BIDSDataLoader>`_
        
        `MVPA model document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.models.html>`_ (Please refer to the corresponding model according to *mvpa_model*.)
        
         Parameters of the above modules can be controlled by input paramter by keywords.
         (e.g. run_mbfmri(..., mask_smoothing_fwhm=6, ..., alpha=0.01) means mask_smoothing_fwhm will be set in VoxelFeatureGenerator and alpha will be set in ElasticNet.)
        
    """
    if analysis.lower() == "mvpa":
        run_mbmvpa(**kwargs)
    elif analysis.lower() == "glm":
        run_mbglm(**kwargs)
    else:
        raise ValueError(f'ERROR: please enter valid analysis type-{valid_analysis}')
        