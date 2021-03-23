from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import bids
from bids import BIDSLayout
from tqdm import tqdm
from .bold_utils import _build_mask, _custom_masking, _image_preprocess
from .bids_utils import BIDSController
import nibabel as nib

from ..utils import config # configuration for default names used in the package


bids.config.set_option("extension_initial_dot", True)

import pdb

class VoxelFeatureGenerator():
    def  __init__(self,
                  bids_layout,
                  bids_controller=None,
                  save_path=None,
                  task_name=None,
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
                  ignore_original=True):
        
        if bids_controller is None:
            self.bids_controller = BIDSController(bids_layout,
                                            save_path=save_path,
                                            fmriprep_name=fmriprep_name,
                                            task_name=task_name,
                                            bold_suffix=bold_suffix,
                                            confound_suffix=confound_suffix,
                                            ignore_original=ignore_original)
        else:
            self.bids_controller = bids_controller
            
        
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
        
        self.mbmvpa_X_suffix = config.DEFAULT_FEATURE_SUFFIX
        
        self.voxel_mask = None
        self.masker = None
        
    def summary(self):
        self.bids_controller.summary()
        
    def _load_voxel_mask(self,overwrite=False):
        if self.bids_controller.voxelmask_path.exists() and not overwrite:
            self.voxel_mask = nib.load(self.bids_controller.voxelmask_path)
        else:
            self.voxel_mask = _build_mask(self.mask_path, self.mask_threshold, self.zoom, verbose=1)
        self.bids_controller.save_voxelmask(self.voxel_mask)    
        
    def run(self,feature_name=None, overwrite=False, confounds=None,n_thread=None):
        
        
        self._load_voxel_mask(overwrite=overwrite)
        t_r = self.bids_controller.meta_infos['t_r'].unique()
        if len(t_r) != 1:
            assert False, "not consistent time resolution"
        t_r = float(t_r[0])
        self.masker = _custom_masking(
                                    self.voxel_mask, t_r,
                                    self.smoothing_fwhm, self.standardize,
                                    self.high_pass, self.detrend
                                    )
        if feature_name is None:
            suffix = self.mbmvpa_X_suffix
        else:
            suffix = feature_name
            
        assert isinstance(suffix, str)
        
        if n_thread is None:
            n_thread = self.n_thread
        assert isinstance(n_thread, int)
        
        if confounds is None:
            confounds = self.confounds
            
        files_layout = []
        skipped_count = 0
        for _, row in self.bids_controller.meta_infos.iterrows():
            
            reg_filename = row['confound_path']
            nii_filename = row['bold_path']
            sub_id = row['subject']
            task_id = row['task']
            ses_id = row['session']
            run_id = row['run']
            
            if ses_id is not None: 
                save_filename = f'sub-{sub_id}_task-{task_id}_ses-{ses_id}_run-{run_id}_{suffix}.npy'
                save_filename = self.bids_controller.set_path(sub_id=sub_id,ses_id=ses_id)/save_filename
            else :
                save_filename = f'sub-{sub_id}_task-{task_id}_run-{run_id}_{suffix}.npy'
                save_filename = self.bids_controller.set_path(sub_id=sub_id)/save_filename
                
            if not overwrite and save_filename.exists():
                skipped_count += 1
                continue
                
            files_layout.append([nii_filename,reg_filename,save_filename, self.confounds, self.masker,self.voxel_mask])
            
        # "chunk_size" is the number of threads.
        # In generalm this number improves performance as it grows,
        # but we recommend less than 4 because it consumes more memory.
        # TODO: We can specify only the number of threads at this time,
        #       but we must specify only the number of cores or both later.
        
        chunk_size = config.MAX_FMRIPREP_CHUNK_SIZE if n_thread > config.MAX_FMRIPREP_CHUNK_SIZE else n_thread
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
        print(f'INFO: start processing fMRI. (nii_img/thread)*(thread)={chunk_size}*{task_size}. {skipped_count} image(s) is(are) skipped.')
        for i, params_chunk in tqdm(enumerate(params_chunks)):
            # parallel computing using multiple threads.
            # please refer to "concurrent" api of Python.
            # it might require basic knowledge in multiprocessing.
            with ProcessPoolExecutor(max_workers=chunk_size) as executor:
                future_result = {executor.submit(
                    _image_preprocess, params): \
                                                params for params in params_chunk
                                }            
                #for future in as_completed(future_result):
        for result in future_result.keys():
            if isinstance(result.exception(),Exception):
                print(future_result[result])
                raise result.exception()
        print(f'INFO: fMRI processing is done.')
        return
        
        