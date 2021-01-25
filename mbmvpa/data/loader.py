#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

from pathlib import Path

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from ..utils import config


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


def BIDSDataLoader():
    
    def __init__(self, layout, 
                 normalizer="minmax",
                 scale=(-1,1),
                 task_name=None, 
                 process_name=None,
                 dynamic_load=False,
                 subjects=None
                ):
         
        if isinstance(layout,str):
            self.layout = BIDSLayout(root=Path(bids_layout),validate=False)
        elif isinstance(layout,Path):
            self.layout = BIDSLayout(root=bids_layout,validate=False)
        elif isinstance(layout,BIDSLayout):
            self.layout = layout
        
        if config.MBMVPA_PIPELINE_NAME in self.layout.derivatives.keys():
            self.layout = self.layout.derivatives[config.MBMVPA_PIPELINE_NAME]
        
        self.task_name=task_name
        self.process_name=process_name
        self.dynamic_load = dynamic_load
        
        if isinstance(normalizer,str):
            self.normalizer = Normalizer(normalizer_name=normalizer_name,
                                         scale=scale)
        else:
            self.normalizer = normalizer
            
        self.X_kwargs = {'suffix':config.DEFAULT_FEATURE_SUFFIX,
                     'extension':'npy'}
        self.y_kwargs = {'suffix':config.DEFAULT_SIGNAL_SUFFIX,
                    'extension':'npy'}
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
        
        if not self.dynamic_load:
            self.X, self.y = self._set_data(self.subjects)
        
    def _get_single_subject_data(self,subject):
        if subject in self.X.keys():
            return X[subject],y[subject]

        subject_Xs =self.layout.get(subject=subject,**self.X_kwargs)

        subject_X_numpy = []
        subject_y_numpy = []

        for subject_X in subject_Xs:
            entities = X_file.get_entities()
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
            subject_X = np.load(subject_X.path)
            subject_y = np.load(subject_y[0].path)
            timemask = np.load(timemask[0].path)==1
            subject_X_numpy.append(subject_X[timemask])
            subject_y_numpy.append(subject_y[timemask])
            
        subject_X_numpy = np.array(subject_X_numpy)
        subject_y_numpy = np.array(subject_y_numpy)
        subject_y_numpy = self.normalizer(subject_y_numpy)    
        
        return subject_X_numpy, subject_y_numpy
    
    
    def _set_data(self,subjects):
        X = {}
        y = {}
        for subject in subjects:
            subject_X_numpy, subject_y_numpy = _get_single_subject_data(subject)
            X[subject]=np.array(subject_X_numpy)
            y[subject]=self.normalizer(np.array(subject_y_numpy))
            
        return X, y
    
    def get_total_data(self):
        X = np.array([self.X[subejct] for subject in self.subjects])
        y = np.array([self.y[subejct] for subject in self.subjects])
        return X,y
            
            