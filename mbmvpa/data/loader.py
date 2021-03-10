#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from bids import BIDSLayout
from ..utils import config
from ..utils.coef2map import reconstruct


class Normalizer():
    def __init__(self, normalizer_name="minmax",scale=(-1,1)):
        self.name = normalizer_name
        self.scale = scale
        
    def __call__(self,x):
        if self.name == "standard":
            normalized = zscore(x, axis=None)
        else:
            # default is using minmax
            original_shape = x.shape
            normalized = minmax_scale(
                x.flatten(),
                feature_range=self.scale, axis=0)
            normalized = normalized.reshape(original_shape)
        return normalized
    
def get_process_name(filename):
    filename = str(filename).split('/')[-1]
    for entity in filename.split('_'):
        entity_key, entity_value =  entity.split('-') 
        if entity_key == config.PROCESS_KEY_NAME:
            return entity_value

    return ""


class BIDSDataLoader():
    
    def __init__(self,
                 layout,
                 voxel_mask_path=None,
                 reconstruct=False,
                 normalizer="minmax",
                 scale=(-1,1),
                 task_name=None, 
                 process_name=None,
                 dynamic_load=False,
                 subjects=None,
                 feature_name=None,
                 loso=False, # leave-one-subject-out
                ):
         
        if isinstance(layout,str) or isinstance(layout,Path):
            try:
                self.layout = BIDSLayout(root=layout,derivatives=True)
            except:
                self.layout = BIDSLayout(root=layout,validate=False)
        elif isinstance(layout,BIDSLayout):
            self.layout = layout
            
        if config.MBMVPA_PIPELINE_NAME in self.layout.derivatives.keys():
            self.layout = self.layout.derivatives[config.MBMVPA_PIPELINE_NAME]
        
        self.task_name=task_name
        self.process_name=process_name
        self.dynamic_load = dynamic_load # Not implemented
        self.reconstruct = reconstruct
        self.loso=loso
        
        
        if feature_name is None:
            self.mbmvpa_X_suffix = config.DEFAULT_FEATURE_SUFFIX
        else:
            self.mbmvpa_X_suffix = feature_name
            
        if voxel_mask_path is None:
            voxel_mask_path = Path(self.layout.root)/ config.DEFAULT_VOXEL_MASK_FILENAME
        
        self.voxel_mask = nib.load(voxel_mask_path)
        
        if isinstance(normalizer,str):
            self.normalizer = Normalizer(normalizer_name=normalizer,
                                         scale=scale)
        else:
            self.normalizer = normalizer
            
        self.X_kwargs = {'suffix':self.mbmvpa_X_suffix,
                     'extension':'npy'}
        self.y_kwargs = {'suffix':config.DEFAULT_SIGNAL_SUFFIX,
                    'extension':'npy'}
        
        self.regressor_kwargs = {'suffix':"regressors",
                          'extension':"tsv"}
        self.timemask_kwargs = {'suffix':config.DEFAULT_TIMEMASK_SUFFIX,
                            'extension':'npy'}
                    
        if self.task_name:
            self.X_kwargs['task']=self.task_name
            self.y_kwargs['task']=self.task_name
            self.timemask_kwargs['task']=self.task_name
        
        
        if subjects is None:
            self.subjects = self.layout.get_subjects()
        else:
            self.subjects = subjects
        self.has_session = len(self.layout.get_sessions())>0
        
        self.X = {}
        self.y = {}
        self.timemask = {}
        self._set_data(self.subjects,self.dynamic_load)
        
            
        
        
    def _get_single_subject_datapath(self,subject):

        subject_Xs =self.layout.get(subject=subject,**self.X_kwargs)

        subject_X_paths = []
        subject_y_paths = []
        timemask_paths = []
        for subject_X in subject_Xs:
            entities = subject_X.get_entities()
            if self.has_session:
                subject_y = self.layout.get(subject=subject,
                                run=entities['run'],
                                session=entities['session'],
                                **self.y_kwargs
                               )
                timemask = self.layout.get(subject=subject,
                                run=entities['run'],
                                session=entities['session'],
                                **self.timemask_kwargs
                               )

            else:
                subject_y = self.layout.get(subject=subject,
                                run=entities['run'],
                                **self.y_kwargs
                               )
                timemask = self.layout.get(subject=subject,
                                run=entities['run'],
                                **self.timemask_kwargs
                               )
                    
            if self.process_name:
                subject_y = [f for f in subject_y if self.process_name==get_process_name(f.filename)]
                timemask = [f for f in timemask if self.process_name==get_process_name(f.filename)]
            
            
            
            if len(subject_y) != 1 or len(timemask) != 1:
                # invalid data layout
                continue
            subject_X_paths.append(subject_X.path)    
            subject_y_paths.append(subject_y[0].path)
            timemask_paths.append(timemask[0].path)
               
        
        return (subject_X_paths,
                subject_y_paths,
                timemask_paths)
    
    
    def _set_data(self,subjects=None,dynamic_load=None,reconstruct=None):
        
            
        self.X = {}
        self.y = {}
        self.timemask = {}
        
        if subjects is None:
            subjects = self.subjects
        if dynamic_load is None:
            dynamic_load = self.dynamic_load
        if reconstruct is None:
            reconstruct = self.reconstruct
            
        for subject in subjects:
            self.X[subject], self.y[subject], self.timemask[subject] = self._get_single_subject_datapath(subject)
        
        
        
        if not dynamic_load:
            for subject in subjects:
                masks = [np.load(f)==1 for f in self.timemask[subject]]
                self.X[subject] = np.concatenate([np.load(f)[masks[i]] for i,f in enumerate(self.X[subject])],0)
                if reconstruct:
                    mask = self.voxel_mask.get_fdata()
                    blackboard = np.zeros(list(mask.shape)+[self.X[subject].shape[0]])
                    blackboard[mask.nonzero()] = self.X[subject].T
                    self.X[subject] = blackboard.T
                self.y[subject] = self.normalizer(np.concatenate([np.load(f)[masks[i]] for i,f in enumerate(self.y[subject])],0))

                self.timemask[subject] = masks
        
    
    def get_data(self, subject_wise=True):
        
            
        if subject_wise:
            return self.X, self.y
        else:
            X = np.concatenate([data for key,data in self.X.items()],0)
            y = np.concatenate([data for key,data in self.y.items()],0)
            return X, y
                
        
    def get_voxel_mask(self):
        return self.voxel_mask