#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.05.03

from .events import LatentProcessGenerator
from .bold import VoxelFeatureGenerator
import numpy as np

class DataPreprocessor():
    r"""
    
    *DataPreprocessor* is for wrapping preprocessing modules for fMRI data and behavioral data.

    The fMRI data preprocessing class, "VoxelFeatureGenerator," is for masking & zooming
    fMRI data ("events.tsv") to make voxel features.
    And, the behavioral data preprocessing class, "LatentProcessGenerator," is for 
    converting behavior data ("events.tsv") to BOLD-like signals.
    You can expect a voxel feature npy file with shape=(time,feature_num) for each run of bold.nii files, 
    and a modulation tsv file with onset, duration, modulation, 
    a time mask npy file, a binary map for indicating valid time points, and a 
    BOLD-like signal npy file for each run of events.tsv files.


    Parameters
    ----------
    
    bids_layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout
        (Original) BIDSLayout of input data. It should follow `BIDS convention`_.
        The main data used from this layout is behaviroal data,``events.tsv``.
    subjects : list of str or "all",default="all"
        List of subject IDs to load. 
        If "all", all the subjects found in the layout will be loaded.
    save_path : str or pathlib.PosixPath, default=None
        Path for saving preprocessed results. The MB-MVPA BIDS-like derivative layout will be created under the given path.
        If not input by the user, it will use "BIDSLayout_ROOT/derivatives/."
    task_name : str, default=None
        Name of the task. If not given, the most common task name will be automatically selected.
    mask_path : str or pathlib.PosixPath, default=None
        Path for directory containing mask files. 
        Mask files are nii files recommended to be downloaded from `Neurosynth`_. 
        As default, each of the nii files is regarded as a probablistic map, and
        the *mask_trheshold* will be used as the cut-off value for binarizing.
        The absolute values are used for thresholding.
        The binarized maps will be integrated by union operation to be a single binary image.
        If None, the default mask_path is 'fmriprep_ROOT/masks.'
    fmriprep_name : str, default="fMRIPrep"
        Name for derivatve BIDS layout for fmriprep.
    bold_suffix : str, default='bold'
        Name of suffix indicating 'bold' image.
        It will be used for searching image files through BIDS layout.
    confound_suffix : str, default='regressors'
        Name of suffix indicating confounds file
        It will be used for searching confounds files through BIDS layout.
    mask_threshold : float, default=2.58
        Cut-off value for thresholding mask images. 
        The default value, 2.58 is determined as it assume the input masks as z maps,
        and 99% confidence interval is considered.
    zoom : tuple[float, float, float], defualt=(2,2,2)
        Window size for zooming fMRI images. Each of three components means x, y ,z axis respectively.
        The size of voxels will be enlarged by the factor of corresponding component value.
        Ex. zoom = (2,2,2) means the original 2 mm^3 voxels will be 4mm^3, so reducing the total number of
        voxels in a single image.
    smoothing_fwhm : float, default=None
        Size in millimeters of the spatial smoothing. If None, skip smoothing.
    standardize : boolean, default=True
        Indicator for standardization.
        If True, Gaussian normalization would be done for each image.
    confounds : list of str, default=[]
        List of names for indicating columns in confounds files.
        The values of *confounds* will be regressed out. 
    high_pass : float, default = 1/128
        Value for high pass filter. [Hz]
    detrend : boolean, default=False
        Indicator for detrending.
        If True, remove a global linear trend in data.
    n_core : int, default=4
        Number of threads for multi-processing. 
        Please consider your computing capcity and enter the affordable number.
        It will be also used as the number of cores for running `hBayesDM`_.
    process_name : str, default="unnamed"
        Name of the target latent process.
        It should be match with the name defined in computational modeling
    adjust_function : function(pandas.Series, dict)-> pandas.Series, default=lambda x : x
        User-defined row-wise function for modifying each row of behavioral data.
        *adjust_function* (a row of DataFrame)-> a row of DataFrame with modified behavior data
    filter_function : function(pandas.Series, dict)-> boolean, default=lambda \_ : True
        User-defined row-wise function for filtering each row of behavioral data.
        *filter_function* (a row of DataFrame)-> True or False
    latent_function : function(pandas.Series, dict)-> pandas.Series, default=None
        User-defined row-wise function for calculating latent process.
        The values will be indexed by 'modulation' column name.
        *latent_function* (a row of DataFrame)-> a row of DataFrame with modulation
    adjust_function_dfwise : function(pandas.DataFrame, dict)-> pandas.DataFrame, default=None
        User-defined dataframe-wise function for modifying each row of behavioral data.
        If not given, it will be made by using *adjust_function*.
        If given, it will override *adjust_function*.
    filter_function_dfwise : function(pandas.DataFrame, dict)-> pandas.DataFrame, default=None
        User-defined dataframe-wise function for filtering each row of behavioral data.
        If not given, it will be made by using *filter_function*.
        If given, it will override *filter_function*.
    latent_function_dfwise : function(pandas.DataFrame, dict)-> pandas.DataFrame, default=None
        User-defined dataframe-wise function for calculating latent process.
        If not given, it will be made by using *latent_function*.
        If given, it will override *latent_function*.
    computational_model : TODO
        TODO
    dm_model : str, default="unnamed"
        Name for computational modeling by `hBayesDM`_. 
        You can still use this parameter to assign the name of the model, 
        even you would not choose to depend on hBayesDM.
    individual_params : str or pathlib.PosixPath or pandas.DataFrame, default=None
        Path or loaded DataFrame for tsv file with individual parameter values.
        If not given, find the file from the default path
        ``MB-MVPA_root/task-*task_name*_model-*model_name*_individual_params.tsv``
        If the path is empty, it will remain ``None`` indicating a need for running hBayesDM.
        So, it will be set after runniing hBayesDM package.
    hrf_model : str, default="glover"
        Name for hemodynamic response function, which will be convoluted with event data to make BOLD-like signal.
        The below notes are retrieved from the code of "nilearn.glm.first_level.hemodynamic_models.compute_regressor"
        (https://github.com/nilearn/nilearn/blob/master/nilearn/glm/first_level/hemodynamic_models.py)

        The different hemodynamic models can be understood as follows:
             - "spm": this is the hrf model used in SPM.
             - "spm + derivative": SPM model plus its time derivative (2 regressors).
             - "spm + time + dispersion": idem, plus dispersion derivative. (3 regressors)
             - "glover": this one corresponds to the Glover hrf.
             - "glover + derivative": the Glover hrf + time derivative (2 regressors).
             - "glover + derivative + dispersion": idem + dispersion derivative. (3 regressors)
    use_duration : boolean, default=False
        Indicator for making time mask. 
        If True use "duration" column to make a time mask, 
        else all the gaps following trials after valid trials would be included in the time mask.    
    ignore_original : boolean, default=False
        Indicator to tell whether it would cover behaviroal data in the original BIDSLayout.
        If True, it will only consider data in the derivative layout for fMRI preprocessed data.
    onset_name : str, default="onset"
        Column name indicating  *onset* values.
    duration_name : str, default="duration"
        Column name indicating *duration* values.
    end_name : str, default=None
        Column name indicating end of valid time.
        If given, *end*-*onset* will be used as *duration* and override *duration_name*.
        If ``None``, it would be ignored and *duration_name* will be used.
    use_1sec_duration : boolean, default=True
        If True, *duration* will be fixed as 1 second.
        This parameter will override *duration_name* and *end_name*.v
                  
      
    .. _`fMRIPrep`: https://fmriprep.org/en/stable/
    .. _`Neurosynth`: https://neurosynth.org/
    .. _`BIDS convention`: https://bids.neuroimaging.io/
    .. _`hBayesDM`: https://hbayesdm.readthedocs.io/en/v1.0.1/models.html
    """
    
    
    def __init__(self,
                  bids_layout,
                  subjects="all",
                  save_path=None,
                  task_name=None,
                  mask_path=None,
                  fmriprep_name='fMRIPrep',
                  bold_suffix='bold',
                  confound_suffix='regressors',
                  mask_threshold=2.58,
                  zoom=(2, 2, 2),
                  smoothing_fwhm=6,
                  standardize=True,
                  confounds=[],
                  high_pass=None,
                  detrend=True,
                  n_core=4,
                  process_name="unnamed",
                  feature_name="unnamed",
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
                  ignore_original=False,
                  onset_name="onset",
                  duration_name="duration",
                  end_name=None,
                  use_1sec_duration=True):
        
        self.X_generator = VoxelFeatureGenerator(bids_layout=bids_layout,
                                                  subjects=subjects,
                                                  save_path=save_path,
                                                  task_name=task_name,
                                                  feature_name=feature_name,
                                                  fmriprep_name=fmriprep_name,
                                                  mask_path=mask_path,
                                                  bold_suffix=bold_suffix,
                                                  confound_suffix=confound_suffix,
                                                  mask_threshold=mask_threshold,
                                                  zoom=zoom,
                                                  smoothing_fwhm=smoothing_fwhm,
                                                  standardize=standardize,
                                                  confounds=confounds,
                                                  high_pass=high_pass,
                                                  detrend=detrend,
                                                  n_thread=n_core,
                                                  ignore_original=ignore_original)
        
        self.bids_controller = self.X_generator.bids_controller
        
        self.y_generator = LatentProcessGenerator(bids_layout=bids_layout,
                                                  subjects=subjects,
                                                  bids_controller=self.bids_controller,
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
                                                  onset_name=onset_name,
                                                  duration_name=duration_name,
                                                  end_name=end_name,
                                                  use_1sec_duration=use_1sec_duration)
    
    def summary(self):
        self.bids_controller.summary()
    
    def preprocess(self, 
                   overwrite=False, 
                   process_name=None,
                   feature_name=None, 
                   confounds=None,
                   n_thread=None,
                   dm_model=None, 
                   individual_params=None, 
                   df_events=None, 
                   adjust_function=None, 
                   filter_function=None,
                   skip_modeling=False,
                   make_plots=True,
                   plot_w=10,
                   plot_h=5,
                   plot_fontsize=12,
                   **kwargs):
            
        
        self.X_generator.run(overwrite=overwrite,
                           feature_name=feature_name,
                           confounds=confounds,
                           n_thread=n_thread,
                            **kwrags)
        
        if not skip_modeling:
            # you can skip fitting hBayesDM
            self.y_generator.set_computational_model(overwrite=overwrite,
                                                    dm_model=dm_model, 
                                                    individual_params=individual_params, 
                                                    df_events=df_events, 
                                                    adjust_function=adjust_function, 
                                                    filter_function=filter_function, 
                                                    **kwargs)
        
        self.y_generator.run(overwrite=overwrite,
                            process_name=process_name)
        
        self.bids_controller.reload()
        
        # make plots of processed data
        if feature_name is None:
            feature_name = self.X_generator.feature_name
        if process_name is None:
            process_name = self.y_generator.process_name
            
        self.bids_controller.plot_processed_data(feature_name=feature_name,
                                                 process_name=process_name,
                                                 w=plot_w,
                                                 h=plot_h,
                                                 fontsize=plot_fontsize)
        
        
        
    