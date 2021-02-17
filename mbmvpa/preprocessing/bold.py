from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import bids
from bids import BIDSLayout
from tqdm import tqdm
from .bold_utils import _custom_masking, _image_preprocess
from .bids_utils import BIDSController
import nibabel as nib

from ..utils import config # configuration for default names used in the package


bids.config.set_option("extension_initial_dot", True)

import pdb

class VoxelFeatureGenerator():
    def  __init__(self,
                  bids_layout,
                  save_path=None,
                  task_name=None,
                  fmriprep_name='fMRIPrep',
                  mask_path=None,
                  bold_suffix='bold',
                  confound_suffix='regressors',
                  mask_threshold=2.58,
                  zoom=(2, 2, 2),
                  smoothing_fwhm=None,
                  interpolation_func=np.mean,
                  standardize=True,
                  confounds=["trans_x", "trans_y",
                                      "trans_z", "rot_x", "rot_y", "rot_z"],
                  high_pass=128,
                  detrend=True,
                  n_thread=4):
        
        self.bids_controller = BIDSController(bids_layout,
                                            save_path=save_path,
                                            fmriprep_name=fmriprep_name,
                                            task_name=task_name,
                                            bold_suffix=bold_suffix,
                                            confound_suffix=confound_suffix)
        
        if mask_path is None:
            self.mask_path = Path(self.bids_controller.fmriprep_layout.root)/ config.DEFAULT_ROI_MASK_DIR
        elif mask_path is False:
            self.mask_path = None
        else:
            self.mask_path = Path(mask_path)
        
        self.confounds = confounds
        self.n_thread = n_thread
        self.voxel_mask, self.masker = _custom_masking(
                                            self.mask_path, mask_threshold, zoom,
                                            smoothing_fwhm, interpolation_func, standardize,
                                            high_pass, detrend
                                        )
        self.mbmvpa_X_suffix = config.DEFAULT_FEATURE_SUFFIX
        self.bids_controller.save_voxelmask(self.voxel_mask)
        
    def summary(self):
        self.bids_controller.summary()
        
    
    def run(self,feature_name=None, overwrite=False, confounds=None,n_thread=None):
        
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
        for file in self.bids_controller.get_bold_all():
            nii_filename = Path(file.dirname)/file.filename
            entities = file.get_entities()
            nii_filedir = file.dirname
            if 'session' in entities.keys():
                reg_file =self.bids_contorller.get_confound(sub_id=entities['subject'],
                                                 ses_id=entities['session'],
                                                 run_id=entities['run'])
                
                reg_file = reg_file[0] # if more than one searched raise warning
                reg_filename = Path(reg_file.dirname)/reg_file.filename
                sub_id, task_id,\
                    ses_id, run_id = entities['subject'], entities['task'],\
                                        entities['session'], entities['run']
                save_filename = f'sub-{sub_id}_task-{task_id}_ses-{ses_id}_run-{run_id}_{suffix}.npy'
                save_filename = self.bids_controller.set_path(sub_id=entities['subject'],ses_id=entities['session'])/save_filename
                if not overwrite and save_filename.exists():
                    continue
            else:
                reg_file=self.bids_controller.get_confound(sub_id=entities['subject'],
                                                 run_id=entities['run'])
                
                reg_file = reg_file[0] # if more than one searched raise warning
                reg_filename = Path(reg_file.dirname)/reg_file.filename
                sub_id, task_id, run_id = entities['subject'], entities['task'], entities['run']
                save_filename = f'sub-{sub_id}_task-{task_id}_run-{run_id}_{suffix}.npy'
                save_filename = self.bids_controller.set_path(sub_id=entities['subject'])/save_filename
                if not overwrite and save_filename.exists():
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
        
        return
        
        