from nilearn.glm.first_level import first_level_from_bids
from nilearn.glm.second_level import SecondLevelModel
import numpy as np
import pandas as pd
from pathlib import Path
from mbmvpa.utils import config
import datetime
from bids import BIDSLayout

class GLM():
    
    def __init__(self,
                 bids_layout,
                 task_name,
                 space_name=None,
                 smoothing_fwhm=6,
                 confounds=['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
                 glm_save_path='.'
                ):
        
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
        self.mbmvpa_layout = layout.derivatives['MB-MVPA']
        self.fmriprep_layout = layout.derivatives['fMRIPrep']
        now = datetime.datetime.now()
        self.save_root = Path(glm_save_path) / f'report_glm_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
        self.save_root.mkdir(exist_ok=True)
        self.save_path_first = self.save_root /'first_map'
        self.save_path_first.mkdir(exist_ok=True)
        self.save_path_second = self.save_root /'second_map'
        self.save_path_second.mkdir(exist_ok=True)
        self.firstlevel_done = False
    
    def run_firstlevel(self):
        
        models, models_run_imgs, \
            models_events, models_confounds = first_level_from_bids(self.layout.root,
                                                                    self.task_name,
                                                                    self.space_name,
                                                                    smoothing_fwhm=self.smoothing_fwhm,
                                                                    derivatives_folder=self.fmriprep_layout.root)
        
        for i in range(len(models_confounds)):
            for j in range(len(models_confounds[i])):
                mc = models_confounds[i][j]
                mc = mc[confounds]
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
                
        first_level_models = [models[i].fit([nib.load(run_img) for run_img in models_run_imgs[i]],
                                            events=models_events[i],
                                            confounds=models_confounds[i]) for i in tqdm(range(len(models)))]
        
        
        for first_level_model in first_level_models:
            contrast_def = [np.zeros( len(dm.columns)) for dm in first_level_model.design_matrices_]
            for i, dm in enumerate(first_level_model.design_matrices_):
                contrast_def[i][dm.columns.get_loc(self.process_name)] = 1
            z_map = first_level_model.compute_contrast(contrast_def=contrast_def,
                                                       output_type='z_score')
            subject_id = first_level_model.subject_label
            nib.save(z_map, self.save_path_first / f'sub-{subject_id}.nii')
            
            
        self.firstlevel_done = True
        
    def run_secondlevel(self):
        
        if not self.firstlevel_done:
            self.run_firstlevel()
        
        second_level_input = [nib.load(nii_file) for nii_file in self.save_path_first.glob('*.nii')]
        design_matrix = pd.DataFrame([1] * len(second_level_input),
                                     columns=['intercept'])
        
        second_level_model = SecondLevelModel(smoothing_fwhm=6.0)
        second_level_model = second_level_model.fit(second_level_input,
                                                    design_matrix=design_matrix)
        
        z_map = second_level_model.compute_contrast(output_type='z_score')
        nib.save(z_map, self.save_path_second / f'secondlevel_z_map.nii')
        
    def run(self):
        self.run_firstlevel()
        self.run_secondlevel()