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
    
    mbglm = MBGLM(config=config,report_path=report_path,**kwargs)
    mbglm.run(overwrite=overwrite,
             overwrite_latent_process=overwrite_latent_process,
             refit_compmodel=refit_compmodel)
    
class MBGLM(MBFMRI):
    
    r"""
        
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
        """
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
    
    def __init__(self,
                 bids_layout,
                 task_name,
                 process_name,
                 fmriprep_layout=None,
                 mbmvpa_layout=None,
                 space_name=None,
                 smoothing_fwhm=6,
                 mask_path=None,
                 mask_threshold=2.58,
                 confounds=['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
                 glm_save_path='.',
                 hrf_model='glover',
                 drift_model='cosine',
                 high_pass=1/128,
                 n_jobs=4,
                 n_core=4,
                 bold_suffix='bold',
                 confound_suffix='regressors',
                 subjects='all'):
        
        # TODO
        # add multi-processing
        # add reporting
        
        self.smoothing_fwhm = smoothing_fwhm
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
            self.fmriprep_layout = self.layout.derivatives['fMRIPrep']
        else:
            self.fmriprep_layout = fmriprep_layout
        now = datetime.datetime.now()
        self.save_root = Path(glm_save_path) / f'report_glm_task-{task_name}_process-{process_name}_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
        self.save_root.mkdir(exist_ok=True)
        self.save_path_first = self.save_root /'first_map'
        self.save_path_first.mkdir(exist_ok=True)
        self.save_path_second = self.save_root /'second_map'
        self.save_path_second.mkdir(exist_ok=True)
        self.firstlevel_done = False
        if mask_path is None:
            self.mask_path = Path(self.fmriprep_layout.root)/'masks'
        else:
            self.mask_path = mask_path
        self.mask_threshold = mask_threshold
        self.hrf_model=hrf_model
        self.drift_model=drift_model
        self.high_pass=high_pass=.01
        self.mask =_build_mask(self.mask_path, self.mask_threshold, (1,1,1), verbose=1)
        self.n_core = n_core
        self.subjects= subjects
        self.bold_suffix = bold_suffix
        self.confound_suffix=confound_suffix
        self.subjects=subjects
        self.confounds = confounds
        self.n_jobs=n_jobs
        
        
    def run_firstlevel(self):
        
        models, models_bold_imgs, \
            models_modulations, models_confounds = first_level_from_bids(self.layout,
                                                                    self.task_name,
                                                                    self.process_name,
                                                                    self.space_name,
                                                                    hrf_model=self.hrf_model,
                                                                    drift_model=self.drift_model,
                                                                    high_pass=self.high_pass,
                                                                    smoothing_fwhm=self.smoothing_fwhm,
                                                                    mask_img = self.mask,
                                                                    bold_suffix=self.bold_suffix,
                                                                    modulation_suffix=config.DEFAULT_MODULATION_SUFFIX,
                                                                    confound_suffix=self.confound_suffix,
                                                                    confound_names=self.confounds,
                                                                    subjects=self.subjects,
                                                                    n_jobs=self.n_jobs
                                                                    #minimize_memory=False,
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
        
        second_level_model = SecondLevelModel(mask_img=self.mask, smoothing_fwhm=6.0)
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
        self.run_firstlevel()
        self.run_secondlevel()