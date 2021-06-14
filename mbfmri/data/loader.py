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
from mbfmri.utils import config
from mbfmri.utils.coef2map import reconstruct
from tqdm import tqdm


class Normalizer():
    r"""
    A callable class for normalizing bold-signals of the latent process. 
    Standardization or Min-Max scaling can be done here for improving 
    stability of MVPA optimization.
    """
    
    def __init__(self, 
                 use_absolute_value=False,
                 normalizer_name="minmax",
                 scale=(-1,1)):
        self.use_absolute_value = use_absolute_value
        self.name = normalizer_name
        self.scale = scale
        
    def __call__(self,x):
        if self.use_absolute_value:
            x = abs(x)
        if self.name == "standard":
            normalized = zscore(x, axis=None)
        elif self.name == "none":
            normalized = x
        else:
            # default is using minmax
            original_shape = x.shape
            normalized = minmax_scale(
                x.flatten(),
                feature_range=self.scale, axis=0)
            normalized = normalized.reshape(original_shape)
        return normalized

class Binarizer():
    
    def __init__(self,
                thresholds=None,
                ratios=.2):
        
        if ratios is not None:
            if isinstance(ratios,float):
                self.upper_ratio,self.lower_ratio = ratios,ratios
            else:
                assert len(ratios) == 2
                self.upper_ratio,self.lower_ratio = ratios
            self.use_ratio = True
        else:
            assert thresholds is not None
            if isinstance(thresholds,float):
                self.upper_thr = thresholds
                self.lower_thr = thresholds
            else:
                self.upper_thr = thresholds[0]
                self.lower_thr = thresholds[1]
            self.use_ratio = False
            
    def __call__(self,x):
        if self.use_ratio:
            d = x.copy().flatten()
            d.sort()
            self.lower_thr = d[int(len(d)*self.lower_ratio)]
            self.upper_thr = d[-int(len(d)*self.upper_ratio)]
        is_positive = (x >= self.upper_thr)
        is_negative = (x < self.lower_thr)
        is_invalid = ~(is_positive|is_negative)
        x[is_positive]=1
        x[is_negative]=0
        x[is_invalid]=-1
        return x
            
    
class BIDSDataLoader():
    
    r"""
    
    BIDSDataLoader is for loading preprocessed fMRI and behaviral data. 
    The files for voxel features, bold-like signals of latent process, time masks,
    and a voxel mask will be aggregated subject-wisely.
    A tensor X with shape (time, voxel_feature_num), and a tensor y with shape (time,)
    will be prepared for each subject. 
    The users can dictionaries for X,y indexed by corresponding subject IDs. 
    Also, temporal masking will be done for each data using time mask file 
    along the time dimension for both of X and y data. 
    Check the codes below for the detail.
    
    Parameters
    ----------
    layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout
        BIDS layout for retrieving MB-MVPA preprocessed data files.
        Users can input the root for entire BIDS layout with original data,
        or the root for MB-MVPA derivative layout.
    subjects : list of str or "all",default="all"
        List of subject IDs to load. 
        If "all", all the subjects found in the layout will be loaded.
    voxel_mask_path : str or pathlib.PosixPath, default=None
        Path for voxel mask file. If None, then find it from default path,
        "MB-MVPA_ROOT/voxelmask_{feature_name}.nii.gz"
    reconstruct : boolean, default=False
        Flag for indicating whether reshape flattened voxel features to
        4D images.
        Normally, it needs to be done for running CNNs.
    normalizer : str, default="none"
        Type of subject-wise normalizaion of bold-like signals of the latent process.
        "none" - do noting.
        "standard" - Gaussian normalization
        "minmax" - Min-Max scaling. linearly rescale to fit in *scale* range.
    scale : tuple[float,float], default=(-1,1)
        Range for "minmax" *normalizer*, otherwise ignored.
    task_name : str, default=None
        Name of the task. 
        If not given, ignored in searching through BIDS layout.
    process_name : str, default="unnamed"
        Name of the target latent process.
        If not given, ignored in searching through BIDS layout.
    feature_name : str, default="unnamed"
        Name for indicating preprocessed feature.
        If not given, ignored in searching through BIDS layout.
    verbose : int, default=1
        Level of verbosity. 
        Currently, if verbose > 0, print all the info. while loading.
    
    """
    
    def __init__(self,
                 layout,
                 subjects='all',
                 voxel_mask_path=None,
                 reconstruct=False,
                 normalizer="none",
                 use_absolute_value=False,
                 scale=(-1,1),
                 task_name=None, 
                 process_name="unnamed",
                 feature_name="unnamed",
                 verbose=1,
                 logistic=False,
                 binarizer_thresholds=None,
                 binarizer_ratios=.2,
                ):
         
        # set MB-MVPA layout from "layout" argument.
        if isinstance(layout,str) or isinstance(layout,Path):
            if len(list(Path(layout).glob('derivatives'))) != 0:
                root_layout = BIDSLayout(root=layout,derivatives=True)
                self.layout = root_layout.derivatives[config.MBMVPA_PIPELINE_NAME]
            else:
                self.layout = BIDSLayout(root=layout,validate=False)
                self.layout.add_derivatives(layout)
        elif isinstance(layout,BIDSLayout):
            if config.MBMVPA_PIPELINE_NAME in layout.derivatives.keys():
                self.layout = layout.derivatives[config.MBMVPA_PIPELINE_NAME]
            else:
                self.layout = layout
                
        # sanity check
        assert self.layout.description['PipelineDescription']['Name'] == config.MBMVPA_PIPELINE_NAME
        
        proces_name = ''.join(process_name.split('_'))
        if verbose > 0:
            print('INFO: retrieving from '+str(self.layout))
            print(f'      task-{task_name}, process-{process_name}, feature-{feature_name}')
            
        self.task_name=task_name
        self.process_name= ''.join(process_name.split('_')) #process_name
        self.feature_name=feature_name
        self.reconstruct = reconstruct
        self.verbose = verbose
        self.nii_ext = config.NIIEXT
        # set & load voxel mask file
        if voxel_mask_path is None:
            voxel_mask_path = Path(self.layout.root)/ f"{config.DEFAULT_VOXEL_MASK_FILENAME}-{feature_name}.{self.nii_ext}"
        
        self.voxel_mask = nib.load(voxel_mask_path)
        
        # initiate normalizer function
        if isinstance(normalizer,str):
            self.normalizer = Normalizer(normalizer_name=normalizer,
                                         use_absolute_value=use_absolute_value,
                                         scale=scale)
        else:
            self.normalizer = normalizer
        
        # set binarizer
        self.logistic = logistic
        if self.logistic:
            self.binarizer = Binarizer(thresholds=binarizer_thresholds,
                                      ratios=binarizer_ratios)
        else:
            self.binarizer = None
        
        # common arguments for retrieving data from BIDS layout
        self.X_kwargs = {'suffix':config.DEFAULT_FEATURE_SUFFIX,
                     'extension':'npy'}
        self.y_kwargs = {'suffix':config.DEFAULT_SIGNAL_SUFFIX,
                    'extension':'npy'}
        self.timemask_kwargs = {'suffix':config.DEFAULT_TIMEMASK_SUFFIX,
                            'extension':'npy'}
        
        # add names in arguments.
        # Names which are not given will be ignored.
        if self.process_name is not None:
            self.y_kwargs['desc'] = self.process_name
            self.timemask_kwargs['desc'] = self.process_name

        if self.feature_name is not None:
            self.X_kwargs['desc'] = self.feature_name
            
        if self.task_name:
            self.X_kwargs['task']=self.task_name
            
        if subjects == 'all':
            self.subjects = self.layout.get_subjects()
        else:
            self.subjects = subjects
        
        self.X = {}
        self.y = {}
        self.timemask = {}
        
        # load data
        self._load_data(self.subjects)
        
    def _get_single_subject_datapath(self,subject):
        
        # find voxel feature, bold-like signals, time mask 
        # data paths for the given subject. 
        
        subject_Xs =self.layout.get(subject=subject,**self.X_kwargs)
        
        subject_X_paths = []
        subject_y_paths = []
        timemask_paths = []
        
        for subject_X in subject_Xs:
            entities = subject_X.get_entities()
            
            if 'session' in entities.keys(): 
                # if session is included in BIDS
                ses_id = entities['session']
            else:
                ses_id = None
                
            if 'run' in entities.keys(): 
                # if run is included in BIDS
                run_id = entities['run']
            else:
                run_id = None
                
            subject_y = self.layout.get(subject=subject,
                            run=run_id,
                            session=ses_id,
                            task=entities['task'],
                            **self.y_kwargs
                           )
            timemask = self.layout.get(subject=subject,
                            run=run_id,
                            session=ses_id,
                            task=entities['task'],
                            **self.timemask_kwargs
                           )

            if len(subject_y) < 1 or len(timemask) < 1:
                # not found. skipped.
                continue
                
            subject_X_paths.append(subject_X.path)    
            subject_y_paths.append(subject_y[0].path)
            timemask_paths.append(timemask[0].path)
               
        return (subject_X_paths,
                subject_y_paths,
                timemask_paths)
    
    def _load_data(self,subjects=None,reconstruct=None):
        
            
        self.X = {}
        self.y = {}
        self.timemask = {}
        
        if subjects is None:
            subjects = self.subjects
        if reconstruct is None:
            reconstruct = self.reconstruct
            
        valid_subjects = []
        
        print('INFO: start loading data')
        iterater = tqdm(subjects, desc='subject', leave=False)
        for subject in iterater:
            iterater.set_description(f"subject_{subject}")
            # get paths for subject's data
            subject_X_paths,subject_y_paths,\
                timemask_paths = self._get_single_subject_datapath(subject)
            
            if len(subject_X_paths) ==0:
                # if not found, skip the subject.s
                continue
            else:
                # add to dictionaries.
                self.X[subject], self.y[subject],\
                    self.timemask[subject] = subject_X_paths, subject_y_paths,\
                                                timemask_paths
                valid_subjects.append(subject)
        
        for subject in valid_subjects:
            # load data X, y and time mask for each subject.
            # mask X, y with time mask.
            # if reconstruct is True, reshape the X data to 4D by using voxel mask. 
            
            masks = [np.load(f)==1 for f in self.timemask[subject]]
            X_subject = [np.load(f) for f in self.X[subject]]
            
            # masked X data are concatenated
            self.X[subject] = np.concatenate([data[mask] for mask,data in zip(masks,X_subject)],0)
            
            if reconstruct:
                mask = self.voxel_mask.get_fdata()
                blackboard = np.zeros(list(mask.shape)+[self.X[subject].shape[0]])
                blackboard[mask.nonzero()] = self.X[subject].T
                self.X[subject] = blackboard.T
                
            # maksed y data are concatenated & normalized 
            self.y[subject] = self.normalizer(np.concatenate([np.load(f)[masks[i]] for i,f in enumerate(self.y[subject])],0))
            
            if self.logistic:
                binarized = self.binarizer(self.y[subject])
                bm = (binarized != -1)
                self.X[subject] =self.X[subject][bm.reshape(*self.X[subject].shape[:-1])]
                self.y[subject] = self.y[subject][bm]
                
            self.timemask[subject] = masks
        
        self.subjects = valid_subjects
        
        if self.verbose > 0:
            # print basic information of loaded data.
            total_size = sum([len(d) for d in self.X])
            print(f'INFO: loaded data info. total-{total_size}')
            for subject in self.subjects:
                X_shape = str(self.X[subject].shape)
                y_shape = str(self.y[subject].shape)
                print(f'      subject_{subject}: X{X_shape}, y{y_shape}')
        
        print(f'INFO: loaded voxel mask{str(self.voxel_mask.get_fdata().shape)}')
        print('INFO: loading data done')
        
    def get_data(self, subject_wise=True):
        
        if subject_wise:
            # return dictionaries of X, y indexed by subject IDs
            return self.X, self.y
        else:
            # Xs, ys from all subjects are concatenated respectively
            X = np.concatenate([data for key,data in self.X.items()],0)
            y = np.concatenate([data for key,data in self.y.items()],0)
            return X, y
                
    def get_voxel_mask(self):
        return self.voxel_mask