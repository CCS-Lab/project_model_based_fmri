#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: heoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.07.23


from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import zscore
from bids import BIDSLayout
from mbfmri.utils import config
from mbfmri.utils.coef2map import reconstruct
from tqdm import tqdm

    
class Normalizer():
    r"""
    Callable class for normalizing bold-signals or latent process signals. 
    Standardization, Rescaling, or Min-Max scaling can be done here for improving 
    stability of MVPA model optimization.
    
    Parameters
    ---------

    normalizer_name : str, default="none"
        Name for the method of normalizaion
        - "none" : do nothing. 
        - "standard" : standardize the data
        - "rescale" : rescale values in to the given scale.
                      the data ranged in [MEAN-std_threshold*STD, MEAN+``std_threshold``*STD] will be
                      resacled to ``scale``. 
        - "minmax" : rescale values in to the given scale (``scale``), so that
                     the entire value range will fit in the scale.
    
    std_threshold : float, default=2.58
        Z-score value for thresholding valid range of data.
        used in "rescale" normalization

    clip : bool, default=True
        Indicate if clip the value range, used in "rescale" normalization.
    
    scale : [float,float], default=[0,1]
        Tuple or list indicating the range, [lower bound, upper bound], for
        the data to be rescaled or fit in.

    use_absolute_value : bool, default=False
        Indicate if use absolute value, use abs(x) instead of input x.


    """
    
    def __init__(self, 
                 normalizer_name="none",
                 std_threshold=2.58,
                 clip=True,
                 scale=[0,1],
                 use_absolute_value=False):

        self.use_absolute_value = use_absolute_value
        self.name = normalizer_name
        self.scale = scale
        self.std_threshold = std_threshold
        self.clip = clip
        
    def __call__(self,x):
        if self.use_absolute_value:
            x = abs(x)
        if self.name == "standard":
            normalized = zscore(x, axis=None)
        elif self.name is None or self.name == "none":
            normalized = x
        elif self.name == "rescale":
            std,mean = x.std(),x.mean()
            ub1,lb1 = mean+self.std_threshold*std,mean-self.std_threshold*std
            lb2,ub2 = self.scale
            normalized = ((x-lb1)*ub2+lb2*(ub1-x))/(ub1-lb1)
            if self.clip:
                normalized=np.clip(normalized, 
                                   a_min=lb2,
                                   a_max=ub2)
        elif self.name == "minmax":
            ub1,lb1 = x.max(), x.min()
            lb2,ub2 = self.scale
            normalized = ((x-lb1)*ub2+lb2*(ub1-x))/(ub1-lb1)
            if self.clip:
                normalized=np.clip(normalized, 
                                   a_min=lb2,
                                   a_max=ub2)
        else:
            normalized = x

        return normalized

class Binarizer():

    r"""
    Callable class for binarizing latent process signals.
    Users can indicate each range for positive (1) or negative (0).
    The datapoints in signals in the positive range will be "1", and
    those in the negative range will be "0." 
    This function is applied for processing target dat for Logistic models. 
    Any siganls with values out of both ranges will be marked as "-1," which
    will be ignored in training the models. 

    Parameters
    ----------

    positive_range : [float,float] or float
        Range, [lower bound, upper bound], for positive logistic value ("1").
        if ``use_ratio`` is True, the bounds in the given range will indicate
        the rates which should be fall into [0,1]. 
        If only one given, [positive_range, 1 or Inf] will be used.

    negative_range : [float,float] or float
        Range, [lower bound, upper bound], for negative logistic value ("0").
        if ``use_ratio`` is True, the bounds in the given range will indicate
        the rates which should be fall into [0,1]. 
        If only one given, [0 or -Inf, negative_range] will be used.

    use_ratio : bool, default=True
        Indicate if boundary values in ranges are in percentage scale.

    """    
    def __init__(self,
                positive_range,
                negative_range,
                use_ratio=True):
        
        self.use_ratio = use_ratio
        
        if isinstance(positive_range,float):
            if self.use_ratio:
                self.positive_range =[positive_range,1.0]
            else:
                self.positive_range =[positive_range,99999]
        else:
            assert len(positive_range) == 2
            self.positive_range =positive_range

        if isinstance(negative_range,float):
            if self.use_ratio:
                self.negative_range =[0,negative_range]
            else:
                self.negative_range =[-99999,negative_range]
        else:
            assert len(negative_range) == 2
            self.negative_range =negative_range
            
    def __call__(self,x):
        if self.use_ratio:
            d = x.copy().flatten()
            d.sort()
            positive_range = [d[int(len(d)*self.positive_range[0])],
                              d[min(len(d)-1,int(len(d)*self.positive_range[1]))]]
            negative_range = [d[max(0,int(len(d)*self.negative_range[0]))],
                              d[int(len(d)*self.negative_range[1])]]
        else:
            positive_range = self.positive_range
            negative_range = self.negative_range
        is_positive = (x >= positive_range[0]) & (x <= positive_range[1])
        is_negative = (x >= negative_range[0]) & (x <= negative_range[1])
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
    
    task_name : str, default=None
        Name of the task. 
        If not given, ignored in searching through BIDS layout.

    process_name : str, default="unnamed"
        Name of the target latent process.
        If not given, ignored in searching through BIDS layout.

    feature_name : str, default="unnamed"
        Name for indicating preprocessed feature.
        If not given, ignored in searching through BIDS layout.

    voxel_mask_path : str or pathlib.PosixPath, default=None
        Path for voxel mask file. If None, then find it from default path,
        "MB-MVPA_ROOT/voxelmask_{feature_name}.nii.gz"

    reconstruct : boolean, default=False
        Flag for indicating whether reshape flattened voxel features to
        4D images.
        Normally, it needs to be done for running CNNs.

    y_normalizer : str, default="none"
        Name for the method of normalizaion of latent process signals (y).
        - "none" : do nothing. 
        - "standard" : standardize the data
        - "rescale" : rescale values in to the given scale.
                      the data ranged in [MEAN-std_threshold*STD, MEAN+``std_threshold``*STD] will be
                      resacled to ``scale``. 
        - "minmax" : rescale values in to the given scale (``scale``), so that
                     the entire value range will fit in the scale.
    
    y_std_threshold : float, default=2.58
        Z-score value for thresholding valid range of latent process signals (y).
        used in "rescale" normalization

    y_clip : bool, default=True
        Indicate if clip the value range, used in "rescale" normalization.
    
    y_scale : [float,float], default=[0,1]
        Tuple or list indicating the range, [lower bound, upper bound], for
        the data to be rescaled or fit in.

    y_use_absolute_value : bool, default=False
        Indicate if use absolute value.

    X_normalizer : str, default="none"
        ``normalizer`` for voxel signals (X).
    
    X_std_threshold : float, default=2.58
        ``std_threshold`` for voxel signals (X).

    X_clip : bool, default=True
        ``clip`` for voxel signals (X).
    
    X_scale : [float,float], default=[0,1]
        ``scale`` for voxel signals (X).

    X_use_absolute_value : bool, default=False
        ``use_absolute_value`` for voxel signals (X).

    logistc : bool, default=False
        Indicate if the data (X,y) is required to be logistic value.
        True when a logistic model is used.

    binarizer_positive_range : [float,float] or float
        Range, [lower bound, upper bound], for positive logistic value ("1").
        if ``use_ratio`` is True, the bounds in the given range will indicate
        the rates which should be fall into [0,1]. 
        If only one given, [positive_range, 1 or Inf] will be used.
        It is valid when ``logistic`` is True.

    binarizer_negative_range : [float,float] or float
        Range, [lower bound, upper bound], for negative logistic value ("0").
        if ``use_ratio`` is True, the bounds in the given range will indicate
        the rates which should be fall into [0,1]. 
        If only one given, [0 or -Inf, negative_range] will be used.
        It is valid when ``logistic`` is True.

    binarizer_use_ratio : bool, default=True
        Indicate if boundary values in ranges are in percentage scale.
        It is valid when ``logistic`` is True.
    
    verbose : int, default=1
        Level of verbosity. 
        Currently, if verbose > 0, print all the info. while loading.
    
    """
    
    def __init__(self,
                 layout,
                 subjects='all',
                 sessions='all',
                 task_name=None, 
                 process_name="unnamed",
                 feature_name="unnamed",
                 voxel_mask_path=None,
                 reconstruct=False,
                 y_normalizer="none",
                 y_scale=[0,1],
                 y_std_threshold=2.58,
                 y_clip=False,
                 y_use_absolute_value=False,
                 X_normalizer="none",
                 X_use_absolute_value=False,
                 X_scale=[0,1],
                 X_std_threshold=2.58,
                 X_clip=False,
                 logistic=False,
                 binarizer_positive_range=None,
                 binarizer_negative_range=None,
                 binarizer_use_ratio=True,
                 verbose=1,
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
        if isinstance(y_normalizer,str):
            self.y_normalizer = Normalizer(normalizer_name=y_normalizer,
                                         use_absolute_value=y_use_absolute_value,
                                         std_threshold=y_std_threshold,
                                         scale=y_scale,
                                          clip=y_clip)
        else:
            self.y_normalizer = y_normalizer
            
        if isinstance(X_normalizer,str):
            self.X_normalizer = Normalizer(normalizer_name=X_normalizer,
                                         use_absolute_value=X_use_absolute_value,
                                         std_threshold=X_std_threshold,
                                         scale=X_scale,
                                          clip=X_clip)
        else:
            self.X_normalizer = X_normalizer
        
        # set binarizer
        self.logistic = logistic
        if self.logistic:
            self.binarizer = Binarizer(positive_range=binarizer_positive_range,
                                       negative_range=binarizer_negative_range,
                                       use_ratio=binarizer_use_ratio)
        else:
            self.binarizer = None
        
        # common arguments for retrieving data from BIDS layout
        self.X_kwargs = {'suffix':config.DEFAULT_FEATURE_SUFFIX,
                     'extension':config.VOXELFTEXT}
        self.y_kwargs = {'suffix':config.DEFAULT_SIGNAL_SUFFIX,
                    'extension':config.MODSIGEXT}
        self.timemask_kwargs = {'suffix':config.DEFAULT_TIMEMASK_SUFFIX,
                            'extension':config.TIMEMASKEXT}
        
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
        
        if sessions =='all':
            sessions = self.layout.get_sessions()
            if len(sessions) ==0:
                has_session = False
            else:
                has_session = True
        else:
            sessions = sessions
            assert len(sessions) > 0
            has_session = True
        if has_session:
            X_kwargs['session'] = sessions
            
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
            self.X[subject] = self.X_normalizer(np.concatenate([data[mask] for mask,data in zip(masks,X_subject)],0))
            
            if reconstruct:
                mask = self.voxel_mask.get_fdata()
                blackboard = np.zeros(list(mask.shape)+[self.X[subject].shape[0]])
                blackboard[mask.nonzero()] = self.X[subject].transpose(1,0)
                self.X[subject] = blackboard.transpose(3,0,1,2)
                
            # maksed y data are concatenated & normalized 
            self.y[subject] = self.y_normalizer(np.concatenate([np.load(f)[masks[i]] for i,f in enumerate(self.y[subject])],0))
            
            # replace NaN with 0
            self.X[subject][np.isnan(self.X[subject])] = 0
            self.y[subject][np.isnan(self.y[subject])] = 0
            
            if self.logistic:
                binarized = self.binarizer(self.y[subject])
                bm = (binarized != -1).flatten()
                self.X[subject] =self.X[subject][bm]
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
                print(f'         subject_{subject}: X{X_shape}, y{y_shape}')
        
        print(f'INFO: loaded voxel mask{str(self.voxel_mask.get_fdata().shape)}')
        print('INFO: loading data done')
        
    def get_data(self, subject_wise=True):
        r"""Get loaded data. 

        Parameters
        ----------

        subject_wise : bool, default=True
            Indicate if the data dictionary indexed by the subject ID 
            is required. If not, a single concatenated array for each will be
            returned.

        """
        if subject_wise:
            # return dictionaries of X, y indexed by subject IDs
            return self.X, self.y
        else:
            # Xs, ys from all subjects are concatenated respectively
            X = np.concatenate([data for key,data in self.X.items()],0)
            y = np.concatenate([data for key,data in self.y.items()],0)
            return X, y
                
    def get_voxel_mask(self):
        r"""Get loaded voxel mask
        """
        return self.voxel_mask