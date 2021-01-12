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
                  n_core=2,
                  n_thread=2):
        
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
        
        self.bids_controller.save_voxelmask(self.voxel_mask)
        
    def _get_preprocess_params(self, motion_confounds=None):
        if motion_confounds is None:
            motion_confounds = self.motion_confounds
        params = []
        subjects = self.bids_controller.get_subjects()
        for subject in subjects:
            # get a list of nii file paths of fMRI images spread in BIDS layout
            nii_layout = self.bids_controller.get_boldfiles(subject)
            # get a list of tsv file paths of regressors spread in BIDS layout
            # e.g. tsv file with motion confound parameters. 
            reg_layout = self.bids_controller.get_confoundiles(subject)

            param = [nii_layout, reg_layout, motion_confounds,
                     self.masker, self.voxel_mask, subject]
            params.append(param)

        assert len(params) == n_subject, (
            "The length of params list and number of subjects are not validated."
        )
        
        return params
    
        
        
        
        