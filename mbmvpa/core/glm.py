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
from mbmvpa.utils import config
from mbmvpa.utils.glm_utils import first_level_from_bids
from mbmvpa.utils.bold_utils import _build_mask
from mbmvpa.preprocessing.events import LatentProcessGenerator
import datetime
from bids import BIDSLayout
from tqdm import tqdm
import nibabel as nib
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib

def run_mbglm(**kwargs):
    mbglm = MBGLM(**kwargs)
    mbglm.run()

class MBGLM():
    
    def __init__(self,
                  bids_layout,
                  task_name,
                  process_name,
                  subjects="all",
                  save_path=None,
                  adjust_function=lambda x: x,
                  filter_function=lambda _: True,
                  latent_function=None,
                  adjust_function_dfwise=None,
                  filter_function_dfwise=None,
                  latent_function_dfwise=None,
                  computational_model=None,
                  dm_model="unnamed",
                  individual_params=None,
                  hrf_model="glover",
                  use_duration=False,
                  n_core=4,
                  ignore_original=False,
                  ignore_fmriprep=False,
                  onset_name="onset",
                  duration_name="duration",
                  end_name=None,
                  use_1sec_duration=True,
                  skip_compmodel=False,
                  separate_run=False,
                  space_name=None,
                  smoothing_fwhm=6,
                  mask_path=None,
                  mask_threshold=2.58,
                  confounds=['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
                  glm_save_path='.',
                  drift_model='cosine',
                  high_pass=1/128,
                  smoothing_fwhm=6,
                  overwrite=False,
                  **kwargs):
        
    if not skip_compmodel and \
            latent_function is None and\
            latent_function_dfwise is None:
        
        modelling_module = f'mbmvpa.preprocessing.computational_modeling.{dm_model}'
        modelling_module = importlib.import_module(modelling_module)
        latent_function_dfwise = modelling_module.ComputationalModel(process_name)
        
        if process_name in modelling_module.latent_process_onset.keys():
            onset_name = modelling_module.latent_process_onset[process_name]
        
    self.y_generator = LatentProcessGenerator(bids_layout=bids_layout,
                                              subjects=subjects,
                                              bids_controller=None,
                                              save_path=save_path,
                                              task_name=task_name,
                                              process_name=process_name,
                                              adjust_function=adjust_function,
                                              filter_function=filter_function,
                                              latent_function=latent_function,
                                              adjust_function_dfwise=adjust_function_dfwise,
                                              filter_function_dfwise=filter_function_dfwise,
                                              latent_function_dfwise=latent_function_dfwise,
                                              computational_model=computational_model,
                                              dm_model=dm_model,
                                              individual_params=individual_params,
                                              hrf_model=hrf_model,
                                              use_duration=use_duration,
                                              n_core=n_core,
                                              ignore_original=ignore_original,
                                              ignore_fmriprep=ignore_fmriprep,
                                              onset_name=onset_name,
                                              duration_name=duration_name,
                                              end_name=end_name,
                                              use_1sec_duration=use_1sec_duration,
                                              skip_compmodel=skip_compmodel,
                                              separate_run=separate_run,
                                              **kwargs)
    
    
    self.glm = None
    
    def run():
        self.y_generator.run(overwrite=overwrite)
        self.y_generator.bids_controller.reload()

        self.glm = GLM(bids_layout=self.y_generator.bids_controller.layout,
                     task_name=task_name,
                     process_name=process_name,
                     fmriprep_layout=self.y_generator.bids_controller.fmriprep_layout,
                     mbmvpa_layout=self.y_generator.bids_controller.mbmvpa_layout,
                     space_name=space_name,
                     smoothing_fwhm=smoothing_fwhm,
                     mask_path=mask_path,
                     mask_threshold=mask_threshold,
                     confounds=confounds,
                     glm_save_path=glm_save_path,
                     hrf_model=hrf_model,
                     drift_model=drift_model,
                     high_pass=high_pass,
                     n_core=n_core,
                    )
        
        self.glm.run()

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
                 n_core=4,
                ):
        
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
            self.layout = layout
        self.confounds = confounds
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
        
    def run_firstlevel(self):
        
        models, models_run_imgs, \
            models_events, models_confounds = first_level_from_bids(self.layout.root,
                                                                    self.task_name,
                                                                    self.space_name,
                                                                    hrf_model=self.hrf_model,
                                                                    drift_model=self.drift_model,
                                                                    high_pass=self.high_pass,
                                                                    smoothing_fwhm=self.smoothing_fwhm,
                                                                    mask_img = self.mask,
                                                                    derivatives_folder=self.fmriprep_layout.root)
        
        for i in range(len(models_confounds)):
            for j in range(len(models_confounds[i])):
                mc = models_confounds[i][j]
                mc = mc[self.confounds]
                models_confounds[i][j] = mc
                
        def get_entity(img_path):
            filename = Path(img_path).stem
            entity = {}
            for z in filename.split('_'):
                if '-' in z:
                    key,val = z.split('-')
                    entity[key] = val
            return entity
        
        for i in range(len(models_run_imgs)):
            for j in range(len(models_run_imgs[i])):
                entity = get_entity(models_run_imgs[i][j])
                kwargs = {}
                if 'ses' in entity.keys():
                    kwargs['session'] = entity['ses']
                if 'run' in entity.keys():
                    kwargs['run'] = entity['run']
                kwargs['subject'] = entity['sub']
                kwargs['task'] = self.task_name
                kwargs['desc'] = self.process_name
                kwargs['suffix'] = config.DEFAULT_MODULATION_SUFFIX #'modulation'
                md =self.mbmvpa_layout.get(**kwargs)[0]
                md = pd.read_table(md)
                md['trial_type'] = [self.process_name]*len(md)
                models_events[i][j] = md
                
                
        def fit_model(params):
            
            models, models_run_imgs, models_events, models_confounds = params
            models.fit([nib.load(run_img) for run_img in models_run_imgs],
                      events=models_events,
                      confounds=models_confounds)
        
        params_chunks = [[[models[i],
                          models_run_imgs[i],
                          models_events[i],
                          models_confounds[i]] for i in range(j,j+self.n_core)]
                            for j in range(0, len(models), self.n_core)]
        future_result = {}
        
        print(f'INFO: start processing {item_count} fMRI. (nii_img/thread)*(n_thread)={len(params_chunks)}*{self.n_core}.')
        
        iterater = tqdm(range(len(params_chunks)))
        for i in iterater:
            iterater.set_description(f"[{i+1}/{len(params_chunks)}]")
            params_chunk = params_chunks[i]
            # parallel computing using multiple threads.
            # please refer to "concurrent" api of Python.
            # it might require basic knowledge in multiprocessing.
            with ProcessPoolExecutor(max_workers=self.n_core) as executor:
                future_result = {executor.submit(
                    fit_model, params): params for params in params_chunk
                                }            
            # check if any error occured.
            for result in future_result.keys():
                if isinstance(result.exception(),Exception):
                    raise result.exception()
                    
            
        '''                 
        first_level_models = [models[i].fit([nib.load(run_img) for run_img in models_run_imgs[i]],
                                            events=models_events[i],
                                            confounds=models_confounds[i]) for i in tqdm(range(len(models)))]
        '''
        
        first_level_models = models
        
        for first_level_model in first_level_models:
            contrast_def = [np.zeros( len(dm.columns)) for dm in first_level_model.design_matrices_]
            for i, dm in enumerate(first_level_model.design_matrices_):
                contrast_def[i][dm.columns.get_loc(self.process_name)] = 1
            z_map = first_level_model.compute_contrast(contrast_def=contrast_def,
                                                       output_type='z_score')
            subject_id = first_level_model.subject_label
            nib.save(z_map, self.save_path_first / f'sub-{subject_id}.nii')
            
        
        self.firstlevel_done = True
        print(f'INFO: first-level analysis is done.')
        
    def run_secondlevel(self):
        
        if not self.firstlevel_done:
            self.run_firstlevel()
        
        second_level_input = [nib.load(nii_file) for nii_file in self.save_path_first.glob('*.nii')]
        
        if len(second_level_input) == 0:
            print("INFO: only one first-level map is found.")
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
        nib.save(z_map, self.save_path_second / f'secondlevel_z_map.nii')
        print("INFO: second-level map is created and saved.")
        
    def run(self):
        self.run_firstlevel()
        self.run_secondlevel()