from concurrent.futures import ProcessPoolExecutor, as_completed
from .event_utils import _get_metainfo
from pathlib import Path
import numpy as np
import bids
from bids import BIDSLayout
from tqdm import tqdm
from .bold_utils import _custom_masking, _image_preprocess_multithreading
from .bids_utils import BIDSController
import nibabel as nib

from ..utils import config # configuration for default names used in the package


bids.config.set_option("extension_initial_dot", True)



class VoxelDataGenerator():
    
    def  __init__(self,
                  bids_layout,
                  save_path=None,
                  fmriprep_name='fMRIPrep',
                  task_name=None,
                  mask_path=None,
                  bold_suffix=None,
                  confound_suffix=None,
                  mask_threshold=2.58,
                  zoom=(2, 2, 2),
                  smoothing_fwhm=6,
                  interpolation_func=np.mean,
                  standardize=True,
                  motion_confounds=["trans_x", "trans_y",
                                      "trans_z", "rot_x", "rot_y", "rot_z"],
                  n_core=2):
        
        self.bids_controller = BIDSController(bids_layout,
                                            save_path=save_path,
                                            fmriprep_name=fmriprep_name,
                                            task_name=task_name,
                                            bold_suffix=bold_suffix,
                                            confound_suffix=confound_suffix)
        
        if mask_path is None:
            self.mask_path = Path(self.bids_controller.fmriprep_layout.root)/ config.DEFAULT_ROI_MASK_DIR
        else:
            self.mask_path = Path(mask_path)
        
        self.motion_confounds = motion_confounds
        self.n_core = n_core
        self.n_thread = n_thread
        self.voxel_mask, self.masker = _custom_masking(
                                            self.mask_path, mask_threshold, zoom,
                                            smoothing_fwhm, interpolation_func, standardize
                                        )
        self.mbmvpa_X_suffix = config.DEFAULT_FEATURE_SUFFIX
        self.bids_controller.save_voxelmask(self.voxel_mask)
        
    def preprocess(self,n_thread=None):
        
        if n_thread is None:
            n_thread = self.n_thread
            
        if motion_confounds is None:
            motion_confounds = self.motion_confounds
        params = []
        get_boldfiles
        
        files_layout = []
        
        for file in self.bids_controller.get_bold():
            nii_filename = file.filename
            entities = file.get_entities()
            if 'session' in entities.keys():
                reg_filename=self.bids_contorller.get_confound(sub_id=entities['subject'],
                                                 ses_id=entities['session'],
                                                 run_id=entities['run'])[0].filename
                save_filename = f'sub-{entities['subject']}_task-{entities['task']}_ses-{entities['session']}_run-{entities['run']}_{self.mbmvpa_X_suffix}.npy'
                save_filename = self.bids_controller.set_path(sub_id=entities['subject'],ses_id=entities['session'])/save_filename
            else:
                reg_filnamee=self.bids_contorller.get_confound(sub_id=entities['subject'],
                                                 run_id=entities['run'])[0].filename
                save_filename = f'sub-{entities['subject']}_task-{entities['task']}_run-{entities['run']}_{self.mbmvpa_X_suffix}.npy'
                save_filename = self.bids_controller.set_path(sub_id=entities['subject'])/save_filename
            
            files_layout.append([nii_filename,reg_filename,save_filename, self.motion_confounds, self.masker,self.voxel_mask])
            
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
        X = []
        for i, params_chunk in enumerate(params_chunks):
            # parallel computing using multiple threads.
            # please refer to "concurrent" api of Python.
            # it might require basic knowledge in multiprocessing.
            with ProcessPoolExecutor(max_workers=chunk_size) as executor:
                for param in params_chunk:
                    executor.submit(_image_preprocess_multithreading, param, chunk_size)
        
        return
        
        