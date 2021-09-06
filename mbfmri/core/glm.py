#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.06.02

r'''

model-based fMRI - GLM version

referenece

    - https://nilearn.github.io/modules/generated/nilearn.glm.second_level.SecondLevelModel.html
    - https://nilearn.github.io/auto_examples/05_glm_second_level/plot_proportion_activated_voxels.html#sphx-glr-auto-examples-05-glm-second-level-plot-proportion-activated-voxels-py
    - https://nilearn.github.io/auto_examples/05_glm_second_level/plot_second_level_one_sample_test.html#sphx-glr-auto-examples-05-glm-second-level-plot-second-level-one-sample-test-py
    - https://nilearn.github.io/modules/generated/nilearn.glm.first_level.first_level_from_bids.html
    - https://nilearn.github.io/auto_examples/07_advanced/plot_bids_analysis.html#sphx-glr-auto-examples-07-advanced-plot-bids-analysis-py
    
'''
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
import pandas as pd
from pathlib import Path
from mbfmri.utils import config
from mbfmri.utils.glm_utils import first_level_from_bids, _fit_firstlevel_model
from mbfmri.utils.bold_utils import _build_mask
from mbfmri.preprocessing.events import LatentProcessGenerator
from mbfmri.core.base import MBFMRI
import mbfmri.utils.config
from mbfmri.utils.plot import *
import yaml, importlib, copy, datetime
from bids import BIDSLayout
from tqdm import tqdm
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed

def run_mbglm(config=None,
              report_path='.',
              overwrite=False,
              overwrite_latent_process=True,
              refit_compmodel=False,
              **kwargs):
    
    r"""
    Callable function of GLM approach of model-based fMRI to enable a single line usage.

    1. preprocess behavioral data to generate latent process signals.
    2. load fMRI images and latent process signals.
    3. run first-level and second-level GLM.

    Parameters
    ----------
    
    config : dict or str or pathlib.PosixPath, default=None
        Dictionary for keyworded configuration, or path for yaml file.
        The configuration input will override the default configuration.
    
    report_path : str or pathlib.PosixPath, defualt="."
        Path for saving outputs.
    
    overwrite : bool, default=False
        Indicate if processing multi-voxel signals is required
        though the files exist.
    
    overwrite_latent : bool, default=False
        Indicate if generating latent process signals is required
        though the files exist.
        
    refit_compmodel : bool, default=False
        Indicate if fitting computational model is required
        though the fitted results (indiv. params. and LOOIC) exist.
        
    **kwargs : dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
        `Generating multi-voxel signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.bold>`_
        
        `Generating latent process signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.events>`_
        
        `Data loading document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.data.html#mbfmri.data.loader.BIDSDataLoader>`_
        
        `GLM document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.core.html#mbfmri.core.glm.GLM>`_ 
        
        Parameters of the above modules can be controlled by input paramter by keywords.
        (e.g. run_mbfmri(..., mask_smoothing_fwhm=6, ..., alpha=0.01) means mask_smoothing_fwhm will be set in VoxelFeatureGenerator and alpha will be set in ElasticNet.)
        Please check `full list of configuration parameters <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.core.html#full-list-of-configuration>_`.
        
    Examples
    --------
    .. code:: python
    
        from mbfmri.core.glm import run_mbglm
        import hbayesdm

        _ = run_mbglm( bids_layout='mini_bornstein2017',    # data
                       dm_model= 'banditNarm_lapse_decay',  # computational model
                       feature_name='zoom2rgrout',          # indentifier for processed fMRI data
                       task_name='multiarmedbandit',        # identifier for task
                       process_name='PEchosen',             # identifier for target latent process
                       subjects='all',                      # list of subjects to include
                       report_path=report_path,             # save path for reporting results
                       confounds=["trans_x", "trans_y",     # list of confounds to be controlled in GLM
                                  "trans_z", "rot_x",
                                  "rot_y", "rot_z"],
                       n_core=4,                            # number of core for multi-processing in hBayesDM
                       n_thread=4,                          # number of thread for multi-threading in generating voxel features
                       overwrite=True,                      # indicate if re-run and overwriting are required
                       refit_compmodel=True,                # indicate if refitting comp. model is required
                      )
    
    """
    
    mbglm = MBGLM(config=config,report_path=report_path,**kwargs)
    mbglm.run(overwrite=overwrite,
             overwrite_latent_process=overwrite_latent_process,
             refit_compmodel=refit_compmodel)
    
class MBGLM(MBFMRI):
    
    r"""
    Class for GLM approach of model-based fMRI to enable a single line usage.

    1. preprocess behavioral data to generate latent process signals.
    2. load fMRI images and latent process signals.
    3. run first-level and second-level GLM.

    Parameters
    ----------
    
    config : dict or str or pathlib.PosixPath, default=None
        Dictionary for keyworded configuration, or path for yaml file.
        The configuration input will override the default configuration.
    
    report_path : str or pathlib.PosixPath, defualt="."
        Path for saving outputs.
    
    **kwargs : dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
        `Generating multi-voxel signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.bold>`_
        
        `Generating latent process signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.events>`_
        
        `Data loading document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.data.html#mbfmri.data.loader.BIDSDataLoader>`_
        
        `GLM document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.core.html#mbfmri.core.glm.GLM>`_ 
        
        Parameters of the above modules can be controlled by input paramter by keywords.
        (e.g. run_mbfmri(..., mask_smoothing_fwhm=6, ..., alpha=0.01) means mask_smoothing_fwhm will be set in VoxelFeatureGenerator and alpha will be set in ElasticNet.)
        Please check `full list of configuration parameters<https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.core.html#full-list-of-configuration>_`.
    
    """
    def __init__(self,
                 config=None,
                 report_path='.',
                 **kwargs):
        
        # load & set configuration
        self.config = self._load_default_config()
        self._override_config(config)
        self._add_kwargs_to_config(kwargs)
        self.config['GLM']['glm_save_path']=report_path
        
        # initiating internal modules for preprocessing input data
        self.y_generator = LatentProcessGenerator(**self.config['LATENTPROCESS'])
        self.config['APPENDIX'] = {}
        self.glm = None
    
    def run(self,
            overwrite=False,
            overwrite_latent_process=True,
            refit_compmodel=False):
        """run model-based fMRI 
        
        Parameters
        ----------

        overwrite : bool, default=False
            Indicate if processing multi-voxel signals is required
            though the files exist.

        overwrite_latent : bool, default=False
            Indicate if generating latent process signals is required
            though the files exist.

        refit_compmodel : bool, default=False
            Indicate if fitting computational model is required
            though the fitted results (indiv. params. and LOOIC) exist.

        """
        
        # y (latent process): comp. model. & hrf convolution
        self.config['HBAYESDM']['refit_compmodel']=refit_compmodel
        self.y_generator.run(modeling_kwargs=self.config['HBAYESDM'],
                            overwrite=overwrite|overwrite_latent_process) 
        self.config['APPENDIX']['best_model'] = self.y_generator.best_model
        # reload bids layout and plot processed data
        self.y_generator.bids_controller.reload()
        
        self.glm = GLM(bids_layout=self.y_generator.bids_controller.layout,
                     fmriprep_layout=self.y_generator.bids_controller.fmriprep_layout,
                     mbmvpa_layout=self.y_generator.bids_controller.mbmvpa_layout,
                     **self.config['GLM'])
        
        self.glm.run()
        
        # save configuration
        save_config_path = str(self.glm.save_root / 'config.yaml')
        yaml.dump(self._copy_config(),open(save_config_path,'w'),indent=4, sort_keys=False)
        
    

class GLM():
    
    """Class for model-based fMRI analysis using GLM approach.
    
    Parameters
    ----------
    
    bids_layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout or BIDSController
        Root for input data. It should follow **BIDS** convention.
    
    task_name : str, default=None
        Name of the task. If not given, the most common task name will be automatically selected.
    
    process_name : str, default="unnamed"
        Name of the target latent process.
        It should be match the name defined in computational modeling
        
    fmriprep_name : str, default="fMRIPrep"
        Name of the derivative layout of fMRI preprocessed data.
    
    fmriprep_layout : bids.layout.layout.BIDSLayout, default=None
        Derivative layout for fMRI preprocessed data. 
        ``fmriprep_layout`` is holding primarily preprocessed fMRI images (e.g. motion corrrected, registrated,...) 
        This package is built upon **fMRIPrep** by *Poldrack lab at Stanford University*. 
        If None, bids_layout.derivatives[fmriprep_name] will be used.
        
    mbmvpa_layout : ids.layout.layout.BIDSLayout, default=None
        Derivative layout for MB-MVPA. 
        The preprocessed voxel features and modeled latent process will be organized within this layout.
        If None, bids_layout.derivatives['MB-MVPA'] will be used.
        
    space_name : str, default=None
        Name of template space. If not given, the most common space in 
        input layout will be selected. 
       
    mask_path : str or pathlib.PosixPath, default=None
        Path for directory containing mask files. 
        If None, the default mask_path is 'BIDS_ROOT/masks.'
        Then, images in mask_path/include will be used to create a mask to include and 
        images in mask_path/exclude will be used to create a mask to exclude
        Mask files are nii files recommended to be downloaded from **Neurosynth**.
        (https://neurosynth.org/)
        As default, each of the nii files is regarded as a probablistic map, and
        the *mask_trheshold* will be used as the cut-off value for binarizing.
        The absolute values are used for thresholding.
        The binarized maps will be integrated by union operation to be a single binary image.
   
    mask_threshold : float, default=1.65
        Cut-off value for thresholding mask images. 
        The default value (=1.65) means the boundary of 
        upper 90% in normal distribution.

    mask_smoothing_fwhm : float, default=6
        Size in millimeters of the spatial smoothing of mask images.
        If None, smoothing is skipped.

    include_default_mask : bool, default=True
        Indicate if the default mask (mni space) should be applied.

    gm_only : bool, default=False
        Indicate if gray matter mask should be applied

    atlas : str, default=None
        Name of atlas when masking by ROIs.
        #TODO add link for list 

    rois : list of str, default=[]
        Names or ROI when masking by ROI.
        #TODO add link for list 

    zoom : (int,int,int),  default=(1,1,1)
        Window size for zooming fMRI images. Each of three components means x, y ,z axis respectively.
        The size of voxels will be enlarged by the factor of corresponding component value.
        Ex. zoom = (2,2,2) means the original 2 mm^3 voxels will be 4mm^3, so reducing the total number of
        voxels in a single image.
    
    glm_save_path : str or pathlib.PosixPath, defualt="."
        Path for saving outputs.
    
    n_core : int, default=4
        Number of core.
    
    bold_suffix : str, default='regressors'
        Name of suffix indicating preprocessed fMRI file
        
    confound_suffix : str, default='regressors'
        Name of suffix indicating confounds file
    
    subjects : list of str or "all", default="all"
        List of valid subject IDs. 
        If "all", all the subjects found in the layout will be loaded.
        
    sessions : list of str or "all", default="all"
        List of valid session IDs. 
        If "all", all the sessions found in the layout will be loaded.
    
    confounds : list of str, default=[]
        Names of confound factors to be regressed out.
        Each should be in the columns of confound files.
        
    t_r : float, default=None
        Time resolution in second. 
        It will be overrided by value from input data if applicable.
    
    slice_time_ref: float, default=.5
        Slice time reference in ratio in 0,1].
        It will be overrided by value from input data if applicable.

    **glm_kwargs : dict
        Dictionary for keywarded arguments for calling *first_level_from_bids* function.
        
    """
    
    def __init__(self,
                 bids_layout,
                 task_name,
                 process_name,
                 fmriprep_name='fMRIPrep',
                 fmriprep_layout=None,
                 mbmvpa_layout=None,
                 space_name=None,
                 mask_path=None,
                 mask_threshold=2.58,
                 mask_smoothing_fwhm=6,
                 include_default_mask=True,
                 atlas=None,
                 rois=[],
                 gm_only=False,
                 glm_save_path='.',
                 n_core=4,
                 bold_suffix='bold',
                 confound_suffix='regressors',
                 subjects='all',
                 sessions='all',
                 zoom=(1,1,1),
                 confounds=[],
                 t_r = None,
                 slice_time_ref=.5,
                 **glm_kwargs):
        
        
        self.task_name = task_name
        self.process_name = process_name
        if isinstance(bids_layout,str) or isinstance(bids_layout,Path): 
            # input is given as path for bids layout root.
            self.layout = BIDSLayout(root=Path(bids_layout),derivatives=True)
        else:
            self.layout = bids_layout
        if space_name is not None:
            self.space_name = space_name
        else:
            self.space_name = config.TEMPLATE_SPACE
            
        if mbmvpa_layout is None:
            self.mbmvpa_layout = self.layout.derivatives['MB-MVPA']
        else:
            self.mbmvpa_layout = mbmvpa_layout
        if fmriprep_layout is None:
            self.fmriprep_layout = self.layout.derivatives[fmriprep_name]
        else:
            self.fmriprep_layout = fmriprep_layout
        now = datetime.datetime.now()
        self.save_root = Path(glm_save_path) / 'glm'
        self.save_root.mkdir(exist_ok=True)
        existing_reports = [-1] + [int(f.split('report-')[-1]) for self.save_root.glob('report-*')]
        existing_reports.sort()
        report_idx = existing_reports[-1] + 1
        self.save_root = self.save_root / f'report-{report_idx}'
        self.save_root.mkdir(exist_ok=True)
        self.save_path_first = self.save_root /'first_map'
        self.save_path_first.mkdir(exist_ok=True)
        self.save_path_second = self.save_root /'second_map'
        self.save_path_second.mkdir(exist_ok=True)
        self.firstlevel_done = False
        if mask_path is None:
            self.mask_path = Path(self.layout.root)/'masks'
        else:
            self.mask_path = mask_path
        self.mask_threshold = mask_threshold
        self.mask_smoothing_fwhm = mask_smoothing_fwhm
        self.include_default_mask = include_default_mask
        self.atlas = atlas
        self.rois = rois
        self.gm_only = gm_only
        self.zoom = zoom
        self.mask = _build_mask(mask_path=self.mask_path,
                              threshold=self.mask_threshold,
                              zoom=self.zoom,
                              smoothing_fwhm=self.mask_smoothing_fwhm,
                              include_default_mask=self.include_default_mask,
                              atlas = self.atlas,
                              rois=self.rois,
                              gm_only=self.gm_only,
                              verbose=1)
        self.n_core = n_core
        self.subjects= subjects
        self.bold_suffix = bold_suffix
        self.confound_suffix=confound_suffix
        self.subjects=subjects
        self.sessions=sessions
        self.glm_kwargs=glm_kwargs
        self.smoothing_fwhm  = glm_kwargs['smoothing_fwhm']
        self.confounds = confounds
        self.slice_time_ref = slice_time_ref
        self.t_r = t_r
        
    def run_firstlevel(self):
        '''run first-level glm
        '''
        try: 
            t_r = self.layout.get_tr()
        except:
            t_r = self.t_r
            
        models, models_bold_imgs, \
            models_modulations, models_confounds = first_level_from_bids(self.layout,
                                                                    self.task_name,
                                                                    self.process_name,
                                                                    self.space_name,
                                                                    t_r = t_r,
                                                                    mask_img = self.mask,
                                                                    bold_suffix=self.bold_suffix,
                                                                    modulation_suffix=config.DEFAULT_MODULATION_SUFFIX,
                                                                    confound_suffix=self.confound_suffix,
                                                                    subjects=self.subjects,
                                                                    sessions=self.sessions,
                                                                    slice_time_ref=self.slice_time_ref,
                                                                    confound_names=self.confounds,
                                                                    **self.glm_kwargs
                                                                    )
        
        
        params_chunks = [[[models[i],
                          models_bold_imgs[i],
                          models_modulations[i],
                          models_confounds[i],
                          self.process_name,
                          self.save_path_first] for i in range(j,min(len(models),j+self.n_core))]
                            for j in range(0, len(models), self.n_core)]
        future_result = {}
        
        print(f'INFO: start running first-level glm. (nii_img/thread)*(n_thread)={len(params_chunks)}*{self.n_core}.')
        
        iterater = tqdm(range(len(params_chunks)))
        for i in iterater:
            iterater.set_description(f"[{i+1}/{len(params_chunks)}]")
            params_chunk = params_chunks[i]
            # parallel computing using multiple threads.
            # please refer to "concurrent" api of Python.
            # it might require basic knowledge in multiprocessing.
            #_fit_firstlevel_model(params_chunk[0])
            
            with ProcessPoolExecutor(max_workers=self.n_core) as executor:
                future_result = {executor.submit(
                    _fit_firstlevel_model, params): params for params in params_chunk
                                }            
            # check if any error occured.
            for result in future_result.keys():
                if isinstance(result.exception(),Exception):
                    raise result.exception()
            
        self.firstlevel_done = True
        print(f'INFO: first-level analysis is done.')
        
    def run_secondlevel(self):
        '''run second-level glm
        '''
        if not self.firstlevel_done:
            self.run_firstlevel()
        
        second_level_input = [nib.load(nii_file) for nii_file in self.save_path_first.glob('*map.nii')]
        
        if len(second_level_input) <= 1:
            print("INFO: only one or zero first-level map is found.")
            print("      Second-level analysis requires two or more subjects' brain maps.")
            return
        else:
            print(f"INFO: {len(second_level_input)} first-level maps are found.")
        design_matrix = pd.DataFrame([1] * len(second_level_input),
                                     columns=['intercept'])
        
        second_level_model = SecondLevelModel(mask_img=self.mask, smoothing_fwhm=self.smoothing_fwhm)
        second_level_model = second_level_model.fit(second_level_input,
                                                    design_matrix=design_matrix)
        
        z_map = second_level_model.compute_contrast(output_type='z_score')
        img_path = self.save_path_second / f'secondlevel_z_map.nii'
        nib.save(z_map, img_path)
        plot_mosaic(img_path,True, self.save_path_second)
        plot_surface_interactive(img_path,True, self.save_path_second)
        plot_slice_interactive(img_path,True, self.save_path_second)
        print("INFO: second-level map is created and saved.")
        
    def run(self):
        '''run glm
        '''
        self.run_firstlevel()
        self.run_secondlevel()