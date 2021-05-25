#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho, Yedarm Seong
## contact: cjfwndnsl@gmail.com
## last modification: 2021.05.03

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import bids
from bids import BIDSLayout
from tqdm import tqdm
from ..utils.bold_utils import _build_mask, _custom_masking, _image_preprocess
from ..utils.bids_utils import BIDSController
import nibabel as nib

from mbmvpa.utils import config # configuration used in the package


bids.config.set_option("extension_initial_dot", True)

class VoxelFeatureGenerator():
    r"""
    
    *VoxelFeatureGenerator* is for masking fMRI data ("events.tsv") to make voxel features.
    The mask image will be obtained by union of given mask images, 
    which are recommended to be downloaded from Neurosynth.
    The output files will be stored in the derivative BIDS layout for the package.
    Users can expect a voxel feature npy file with shape=(time,feature_num) for each run of bold.nii files.

    Parameters
    ----------
    
    bids_layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout
        (Original) BIDSLayout of input data. It should follow BIDS convention.
        The main data used from this layout is behaviroal data,``events.tsv``.
    subjects : list of str or "all",default="all"
        List of subject IDs to load. 
        If "all", all the subjects found in the layout will be loaded.
    bids_controller : mbmvpa.utils.bids_utils.BIDSController, default=None
        BIDSController instance for controlling BIDS layout for preprocessing.
        If not given, then initiates the controller.
    save_path : str or pathlib.PosixPath, default=None
        Path for saving preprocessed results. The MB-MVPA BIDS-like derivative layout will be created under the given path.
        If not input by the user, it will use "BIDSLayout_ROOT/derivatives/."
    task_name : str, default=None
        Name of the task. If not given, the most common task name will be automatically selected.
    feature_name : str, default="unnamed"
        Name for indicating preprocessed feature.
        This is adopted for distinguishing different configuration of feature processing.
    fmriprep_name : str, default="fMRIPrep"
        Name for derivatve BIDS layout for fmriprep.
        As default, it is "fMRIPrep", which is the name of preprocessing by **fMRIPrep**.
        (https://fmriprep.org/en/stable/)
    mask_path : str or pathlib.PosixPath, default=None
        Path for directory containing mask files. 
        Mask files are nii files recommended to be downloaded from **Neurosynth**.
        (https://neurosynth.org/)
        As default, each of the nii files is regarded as a probablistic map, and
        the *mask_trheshold* will be used as the cut-off value for binarizing.
        The absolute values are used for thresholding.
        The binarized maps will be integrated by union operation to be a single binary image.
        If None, the default mask_path is 'fmriprep_ROOT/masks.'
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
        If smoothing_fwhm is not None, it gives the size in millimeters of the
        spatial smoothing to apply to the signal.
    standardize : boolean, default=True
        If True, standardization (Gaussian normalization) would be done for each image.
    confounds : list of str, default=[]
        The list of names for indicating columns in confounds files.
        The values of *confounds* will be regressed out. 
    high_pass : float, default = 1/128
        Value for high pass filter. [Hz]
    detrend : boolean, default=False
        If True, remove a global linear trend in data.
    n_thread : int, default=4
        Number of threads for multi-processing. 
        Please consider your computing capcity and enter the affordable number.
    ignore_original : boolean, default=True
        Indicator to tell whether it would cover behaviroal data in the original BIDSLayout.
        If True, it will only consider data in the derivative layout for fMRI preprocessed data.
    
    """
    
    def  __init__(self,
                  bids_layout,
                  subjects="all",
                  bids_controller=None,
                  save_path=None,
                  task_name=None,
                  feature_name="unnamed",
                  fmriprep_name='fMRIPrep',
                  mask_path=None,
                  bold_suffix='bold',
                  confound_suffix='regressors',
                  mask_threshold=2.58,
                  zoom=(2, 2, 2),
                  smoothing_fwhm=None,
                  standardize=True,
                  confounds=[],
                  high_pass=1/128,
                  detrend=False,
                  n_thread=4,
                  ignore_original=True,
                  **kwargs):
        
        # set path informations and load layout
        if bids_controller is None:
            self.bids_controller = BIDSController(bids_layout,
                                            subjects=subjects,
                                            save_path=save_path,
                                            fmriprep_name=fmriprep_name,
                                            task_name=task_name,
                                            bold_suffix=bold_suffix,
                                            confound_suffix=confound_suffix,
                                            ignore_original=ignore_original)
        else:
            self.bids_controller = bids_controller
            
        self.bids_controller._set_voxelmask_path(feature_name=feature_name)
        
        if mask_path is None:
            self.mask_path = Path(self.bids_controller.fmriprep_layout.root)/ config.DEFAULT_ROI_MASK_DIR
        elif mask_path is False:
            self.mask_path = None
        else:
            self.mask_path = Path(mask_path)
    
        self.confounds = confounds
        self.n_thread = n_thread
        self.mask_threshold = mask_threshold
        self.zoom = zoom
        self.smoothing_fwhm = smoothing_fwhm
        self.standardize = standardize
        self.high_pass = high_pass
        self.detrend = detrend
        self.feature_name = feature_name
        self.mbmvpa_X_suffix = config.DEFAULT_FEATURE_SUFFIX
        self.voxel_mask = None
        self.masker = None
        
    def summary(self):
        self.bids_controller.summary()
        
    def _load_voxel_mask(self,overwrite=False):
        if self.bids_controller.voxelmask_path.exists() and not overwrite:
            # if a mask file exists, then load it
            # if overwrite is True, then re-make the mask file
            self.voxel_mask = nib.load(self.bids_controller.voxelmask_path)
            # for printing 
            m = self.voxel_mask.get_fdata()
            survived = int(m.sum())
            total = np.prod(m.shape)
            print('INFO: existing voxel mask is loaded.'+f': {survived}/{total}')
        else:
            # integrate mask files in mask_path. 
            self.voxel_mask = _build_mask(self.mask_path, self.mask_threshold, self.zoom, verbose=1)
        # save voxel mask
        self.bids_controller.save_voxelmask(self.voxel_mask)    
        
    def run(self,
            feature_name=None,
            overwrite=False,
            confounds=None,
            n_thread=None,
            **kwargs):
        
        if n_thread is None:
            n_thread = self.n_thread
        assert isinstance(n_thread, int)
        
        n_thread = max(n_thread,1)
        chunk_size = min(n_thread,config.MAX_FMRIPREP_CHUNK_SIZE)
        
        # upperbound for n_thread is MAX_FMRIPREP_CHUNK_SIZE
        # check utils/config.py
        # "chunk_size" is the number of threads.
        # In generalm this number improves performance as it grows,
        # but we recommend less than 4 because it consumes more memory.
        # TODO: We can specify only the number of threads at this time,
        #       but we must specify only the number of cores or both late
        
        
        
        self._load_voxel_mask(overwrite=overwrite)
        
        t_r = self.bids_controller.meta_infos['t_r'].unique()
        if len(t_r) != 1:
            # check if all the time resolution are same.
            assert False, "not consistent time resolution"
        t_r = float(t_r[0])
        
        # initiate maskers thread-by-thread.
        # so, len(self.maskers) == chunk_size
        # masker masks and zooms each image to get voxel features
        self.maskers = [_custom_masking(
                                    self.voxel_mask, t_r,
                                    self.smoothing_fwhm, self.standardize,
                                    self.high_pass, self.detrend
                                    ) for _ in range(chunk_size)]
        
        if feature_name is None:
            suffix = self.mbmvpa_X_suffix
        else:
            suffix = feature_name
            
        assert isinstance(suffix, str)
        
        
        if confounds is None:
            confounds = self.confounds
            
        # stats
        skipped_count = 0
        item_count = 0
        
        files_layout = []

        for _, row in self.bids_controller.meta_infos.iterrows():
            # get and organize input data for running maskers
            reg_filename = row['confound_path']
            nii_filename = row['bold_path']
            sub_id = row['subject']
            task_id = row['task']
            ses_id = row['session']
            run_id = row['run']
            
            save_filename = f'sub-{sub_id}_task-{task_id}'
            if ses_id is not None: 
                save_filename += f'_ses-{ses_id}'
            if run_id is not None:
                save_filename += f'_run-{run_id}'
            save_filename = f'_desc-{self.feature_name}_{suffix}.npy'
            save_filename = self.bids_controller.set_path(sub_id=sub_id,ses_id=ses_id)/save_filename
            
            if not overwrite and save_filename.exists():
                # if the output already exists, skip it.
                skipped_count += 1
                continue
                
            save_filename = str(save_filename)
            
            files_layout.append([nii_filename,
                                 reg_filename,
                                 save_filename, 
                                 self.confounds, 
                                 self.maskers[item_count%chunk_size]])
            item_count += 1
        
        # re-organize input infos chunk-wise
        params_chunks = [files_layout[i:i + chunk_size]
                            for i in range(0, len(files_layout), chunk_size)]
        task_size = len(params_chunks)

        # Parallel processing for images process with process pool
        # Process pool has the advantage of high performance compared to thread pool.
        # 1. Crate process pool - ProcessPoolExecutor
        # 2. Create parameters to use for each task in process - params_chunk
        # 3. Thread returns a return value after job completion - future.result()
        # ref.: https://docs.python.org/ko/3/library/concurrent.futures.html
        
        future_result = {}
        print(f'INFO: start processing {item_count} fMRI. (nii_img/thread)*(n_thread)={task_size}*{chunk_size}. {skipped_count} image(s) is(are) skipped.')
        
        iterater = tqdm(range(task_size))
        for i in iterater:
            iterater.set_description(f"[{i+1}/{task_size}]")
            params_chunk = params_chunks[i]
            # parallel computing using multiple threads.
            # please refer to "concurrent" api of Python.
            # it might require basic knowledge in multiprocessing.
            with ProcessPoolExecutor(max_workers=chunk_size) as executor:
                future_result = {executor.submit(
                    _image_preprocess, params): \
                                                params for params in params_chunk
                                }            
            # check if any error occured.
            for result in future_result.keys():
                if isinstance(result.exception(),Exception):
                    raise result.exception()
                    
        # end of process
        print(f'INFO: fMRI processing is done.')
        return
        
        