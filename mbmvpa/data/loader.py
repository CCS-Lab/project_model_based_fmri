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
                 subjects=None
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
        self.dynamic_load = dynamic_load
        self.reconstruct = reconstruct
        
        if voxel_mask_path is None:
            voxel_mask_path = Path(self.layout.root)/ config.DEFAULT_VOXEL_MASK_FILENAME
        
        self.voxel_mask = nib.load(voxel_mask_path)
        
        if isinstance(normalizer,str):
            self.normalizer = Normalizer(normalizer_name=normalizer,
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
            subject_X = np.load(subject_X.path)
            subject_y = np.load(subject_y[0].path)
            timemask = np.load(timemask[0].path)==1
            subject_X = subject_X[timemask]
            #if self.reconstruct:
                #subject_X = np.array([reconstruct(array, self.voxel_mask.get_fdata()) for array in subject_X])
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
            subject_X_numpy, subject_y_numpy = self._get_single_subject_data(subject)
            X[subject]=np.array(subject_X_numpy)
            y[subject]=self.normalizer(np.array(subject_y_numpy))
            
        return X, y
    
    def get_total_data(self, flatten=True,reconstruct=None):
        X = np.array([self.X[subject] for subject in self.subjects])
        y = np.array([self.y[subject] for subject in self.subjects])
        if flatten:
            X=X.reshape(-1,X.shape[-1])
            y=y.flatten()
            
        if reconstruct is None:
            reconstruct = self.reconstruct
            
        if reconstruct:
            mask = self.voxel_mask.get_fdata()
            blackboard = np.zeros(list(mask.shape)+[X.shape[0]])
            blackboard[mask.nonzero()] = X.T
            X = blackboard.T
            
        return X,y
    
    def get_voxel_mask(self):
        return self.voxel_mask
            

class DataGenerator(Sequence):
    """
    Data generator required for fitting Keras model. This is just a
    simple wrapper of generating preprocessed fMRI data (:math:`X`) and BOLD-like
    target data (:math:`y`).
    
    Please refer to the below links for examples of using DataGenerator for Keras deep learning framework.
        - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        
    Also, this class is used to generate a chunk of data called 'batch', 
    which means a fragment aggregatin the specified number ('batch_size') of data (X,y).
    This partitioning data to small size is intended for utilizing the mini-batch gradient descent (or stochastic gradient descent).
    Please refer to the below link for the framework.
        - https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/stochastic-gd.pdf
    # TODO find a better reference
    """

    def __init__(self, X, y, batch_size, shuffle=True, use_bipolar_balancing=False, binarize=False,**kwargs):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(X.shape[0])
        self.binarize= binarize
        if self.binarize:
            use_bipolar_balancing = True
            if 'high_rate' in kwargs.key():
                high_rate = kwargs['high_rate']
            else:
                high_rate = None
                
            if 'low_rate' in kwargs.key():
                low_rate = kwargs['low_rate']
            else:
                low_rate = None
                
            self.binarizer = get_binarizer(y.flatten(),high_rate,low_rate)
        self.use_bipolar_balancing=use_bipolar_balancing
        if self.use_bipolar_balancing:
            self.ticketer = get_bipolarized_ticketer(y.flatten(),**kwargs)
            self.X_original = X
            self.y_original = y
        
        self.on_epoch_end()

    # for printing the statistics of the function
    def on_epoch_end(self):
        "Updates indexes after each epoch"

        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        if self.use_bipolar_balancing:
            sample_ids = weighted_sampling(self.y_original, self.ticketer)
            self.X = self.X_original[sample_ids]
            self.y = self.y_original[sample_ids]
            
    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        "Get a batch of data X, y"
        # index : batch no.
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        images = [self.X[i] for i in indexes]
        if self.binarize:
            targets = [self.binarizer(self.y[i]) for i in indexes]
        else:
            targets = [self.y[i] for i in indexes]
        images = np.array(images)
        targets = np.array(targets)

        return images, targets  # return batch

def gaussian(x, mean, std):
    return ((2*np.pi*(std**2))**(-.5))*np.exp(-.5*(((x-mean)/std)**2))


    
def get_bipolarized_ticketer(array,high_rate=.1,low_rate=.1, max_val=None, min_val=None, bins=100, max_ticket=10):
    d = array.copy().flatten()
    d.sort()
    low_part = d[:int(len(d)*low_rate)]
    low_part = np.concatenate([low_part,low_part.max()*2 -low_part],0)
    high_part = d[-int(len(d)*high_rate):]
    high_part = np.concatenate([high_part,high_part.min()*2 -high_part],0)
    
    low_mean = low_part.mean()
    low_std = low_part.std()
    
    high_mean = high_part.mean()
    high_std = high_part.std()
    
    if max_val is None:
        max_val = d[-1]
    if min_val is None:
        min_val = d[0]
    
    x = np.linspace(min_val, max_val, bins)
    
    weights = gaussian(x, low_mean, low_std) + gaussian(x, high_mean, high_std)
    weight_max = weights.max()
    ticketer = lambda v: int(((gaussian(v, low_mean, low_std) + \
                              gaussian(v, high_mean, high_std)) /weight_max+1/max_ticket) * max_ticket)
    
    return ticketer

def get_binarizer(array,high_rate=.1,low_rate=.1):
    d = array.copy().flatten()
    d.sort()
    low_pole = d[int(len(d)*low_rate)]
    high_pole = d[-int(len(d)*high_rate)]
        
    binarizer = lambda v: int((high_mean-v)<(v-low_mean))
    
def weighted_sampling(y, ticketer, n_sample=None):
    
    if n_sample is None:
        n_sample = len(y)
    
    pool = []
    
    for i,v in enumerate(y.flatten()):
        pool += [i]*ticketer(v)
    
    sample_ids  = np.random.choice(pool,n_sample)
        
    return sample_ids

def get_binarizing_thresholds(array,high_rate=.1,low_rate=.1, max_val=None, min_val=None, bins=100, max_ticket=10):
    d = array.copy().flatten()
    d.sort()
    low_part = d[:int(len(d)*low_rate)]
    low_part = np.concatenate([low_part,low_part.max()*2 -low_part],0)
    high_part = d[-int(len(d)*high_rate):]
    high_part = np.concatenate([high_part,high_part.min()*2 -high_part],0)
    
    low_mean = low_part.mean()
    low_std = low_part.std()
    
    high_mean = high_part.mean()
    high_std = high_part.std()
    
    if max_val is None:
        max_val = d[-1]
    if min_val is None:
        min_val = d[0]
    
    x = np.linspace(min_val, max_val, bins)
    
    weights = gaussian(x, low_mean, low_std) + gaussian(x, high_mean, high_std)
    weight_max = weights.max()
    ticketer = lambda v: int(((gaussian(v, low_mean, low_std) + \
                              gaussian(v, high_mean, high_std)) /weight_max+1/max_ticket) * max_ticket)
    
    return ticketer