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
    It offers GLM approach and MVPA approach. For each approach,
    the following procedures are executed.
    
    MVPA approach
    1. process fMRI & behavioral data to generate multi-voxel bold signals and latent process signals
    2. load processed signals.
    3. fit MVPA models and interprete the models to make a brain map.
    
    GLM approach
    1. preprocess behavioral data to generate latent process signals.
    2. load fMRI images and latent process signals.
    3. run first-level and second-level GLM.
    
    By running this code, users can expect to get a brain implementation of 
    the target latent process defined in the computational model. 
    
    
    Parameters
    ----------
    
    analysis: str, default="mvpa"
        Name of approach. "mvpa" will conduct MVPA approach by running *run_mbmvpa*,
        and "glm" will conduct GLM approach by running *run_mbglm*.
        
    
    kwargs: dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
        `Generating multi-voxel signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.bold>`_
        
        `Generating latent process signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.events>`_
        
        `Data loading document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.data.html#mbfmri.data.loader.BIDSDataLoader>`_
        
         If analysis == "mvpa", `MVPA model document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.models.html>`_ (Please refer to the corresponding model according to *mvpa_model*.)
         
         If analysis == "glm", `GLM document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.core.html#mbfmri.core.glm.GLM>`_
         
        Parameters of the above modules can be controlled by input paramter by keywords.
        (e.g. run_mbfmri(..., mask_smoothing_fwhm=6, ..., alpha=0.01) means mask_smoothing_fwhm will be set in VoxelFeatureGenerator and alpha will be set in ElasticNet.)
        
        
    Examples
    --------
    .. code:: python
    
        from mbfmri.core.engine import run_mbfmri
        import hbayesdm

        _ = run_mbfmri(analysis='mvpa',                     # name of analysis, "mvpa" or "glm"
                       bids_layout='mini_bornstein2017',    # data
                       mvpa_model='elasticnet',             # MVPA model, "mlp" or "cnn" for DNN
                       dm_model= 'banditNarm_lapse_decay',  # computational model
                       feature_name='zoom2rgrout',          # indentifier for processed fMRI data
                       task_name='multiarmedbandit',        # identifier for task
                       process_name='PEchosen',             # identifier for target latent process
                       subjects='all',                      # list of subjects to include
                       method='5-fold',                     # type of cross-validation
                       report_path=report_path,             # save path for reporting results
                       confounds=["trans_x", "trans_y",     # list of confounds to regress out
                                  "trans_z", "rot_x",
                                  "rot_y", "rot_z"],
                       n_core=4,                            # number of core for multi-processing in hBayesDM
                       n_thread=4,                          # number of thread for multi-threading in generating voxel features
                       overwrite=True,                      # indicate if re-run and overwriting are required
                       refit_compmodel=True,                # indicate if refitting comp. model is required
                      )
        
    """
    if analysis.lower() == "mvpa":
        run_mbmvpa(**kwargs)
    elif analysis.lower() == "glm":
        run_mbglm(**kwargs)
    else:
        raise ValueError(f'ERROR: please enter valid analysis type-{valid_analysis}')
        