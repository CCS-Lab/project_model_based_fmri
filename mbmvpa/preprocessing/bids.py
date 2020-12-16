#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

"""
Input fMRI image data organized in BIDS layout should be preprocessed to 1. be motion corrected, 2. have a reduced dimension and 3. be aggregated subject-wise.

1. **To remove motion artifacts and drift** 

    Done by wrapping functions in *nilearn* package.
    
2. **To reduce dimensionality by masking and pooling**

    It is an important process as the high dimensionality is an obstacle for fitting regression model, in terms of both computing time and convergence.

    2-a. Masking - Related argument (*mask_path*,``mask_path``),(*threshold*,``threshold``)
        Probabilistic maps are integrated to make the maps as ROIs (a binary voxel-wise mask). 
        It is needed to threshold the map so that you create a mask that only includes voxels with a z-score of a specific value greater than the threshold.
        After thresholding, the surviving voxels are binarized - in other words, set to 1.
        The mask extracts the data from voxels within that region (We can extract the voxels from the mask / Only voxel whose mask value is 1 will be  extracted)
        -> This process reduces the total number of voxels that will be included in the analysis.
        If the masking information is not provided, all the voxels in MNI 152 space will be included in the data.

    2-b. Pooling - Related argument (*zoom*,``zoom``), (*interpolation_func*,``interpolation_func``)
        The number of voxels is further diminished by zooming (or resacling) fMRI images to a coarse-grained resolution. 
        You can give a tuple indicating a zooming window size in x,y,z directions. e.g. (2,2,2)
        Voxels in a cube with the zooming window size will be converted to one representative value reducing resolution and the total number of voxels.
        You can also indicate the method to extract representative value with numpy function. e.g. np.mean means using the average value.

3. **To re-organize data for fitting MVPA model** 

    The preprocessed image will be saved subject-wise.
    
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from .event_utils import _get_metainfo
from pathlib import Path
import numpy as np
import bids
from bids import BIDSLayout
from tqdm import tqdm
from .fMRI import _custom_masking, _image_preprocess_multithreading
import nibabel as nib

from ..utils import config # configuration for default names used in the package


bids.config.set_option("extension_initial_dot", True)


def bids_preprocess(# path informations
                    root=None,
                    layout=None,
                    save_path=None,
                    # ROI masking specification
                    mask_path=None,
                    threshold=2.58,
                    # preprocessing specification
                    zoom=(2, 2, 2),
                    smoothing_fwhm=6,
                    interpolation_func=np.mean,
                    standardize=True,
                    motion_confounds=["trans_x", "trans_y",
                                      "trans_z", "rot_x", "rot_y", "rot_z"],
                    ncore=2,
                    nthread=2):
    """Preprocessing fMRI image data organized in BIDS layout.

    What you need as input:
    
    - **fMRI image data** organized in **BIDS** layout 
    
    *Reamark* you need to provide fMRI data which has gone through the conventional primary preprocessing pipeline (recommended link: https://fmriprep.org/en/stable/), and should be in "BIDSroot/derivatives/fmriprep" 
    
    - **mask images** (nii or nii.gz format) either downloaded from Neurosynth or created by the user

    Args:
        root (str or Path) : the root directory of BIDS layout
        layout (bids.BIDSLayout): BIDSLayout by bids package. if not provided, it will be obtained from root path.
        save_path (str or Path): a path for the directory to save output (X, voxel_mask). if not provided, "BIDS root/derivatives/data" will be set as default path      
        mask_path (str or Path): a path for the directory containing mask files (nii or nii.gz). encourage get files from Neurosynth
        threshold (float): threshold for binarizing mask images
        zoom (tuple[float,float,float]): zoom window, indicating a scaling factor for each dimension in x,y,z. the dimension will be reduced by the factor of corresponding axis.
                                    e.g. (2,2,2) will make the dimension half in all directions, so 2x2x2=8 voxels will be 1 voxel.
        smoothing_fwhm (int): the amount of spatial smoothing. if None, image will not be smoothed.
        interpolation_func (numpy.func): a method to calculate a representative value in the zooming window. e.g. numpy.mean, numpy.max
                                         e.g. zoom=(2,2,2) and interpolation_func=np.mean will convert 2x2x2 cube into a single value of its mean.
        standardize (bool): if true, conduct standard normalization within each image of a single run. 
        motion_confounds (list[str]): list of motion confound names in confounds tsv file. 
        ncore (int): the number of core for the tparallel computing 
        nthread (int): the number of thread for the parallel computing
        
    Returns:
        tuple[numpy.array, nibabel.nifti1.Nifti1Image, nilearn.NiftiMasker, nibabel.BIDSLayout]: 
        - **X** (*numpy.ndarray*) - input data for MVPA(:math:`X`). subject-wise & run-wise BOLD time series data. shape : subject # x run # x timepoint # x voxel #
        - **voxel_mask** (*nibabel.nifti1.Nifti1Image*) - a nifti image for voxel-wise binary mask (ROI mask)
        - **masker** (*nilearn.input_data.NiftiMasker*) - the masker object. fitted and used for correcting motion confounds, and masking.
        - **layout** (*bids.BIDSLayout*) - the loaded layout. 
    """

    progress_bar = tqdm(total=6)
    ###########################################################################
    # parameter check

    progress_bar.set_description("checking parameters..".ljust(50))

    # path informations
    from_root = True
    if root is None:
        assert (layout is not None)
        from_root = False

    assert (save_path is None
        or isinstance(save_path, str)
        or isinstance(save_path, Path))

    # ROI masking specification
    assert (mask_path is None
        or isinstance(mask_path, str)
        or isinstance(mask_path, Path))

    assert (isinstance(threshold, float)
        or isinstance(threshold, int))

    # preprocessing specification
    assert (isinstance(zoom, list)
        or isinstance(zoom, tuple))
    assert isinstance(zoom[0], int)

    assert (smoothing_fwhm is None
        or isinstance(smoothing_fwhm, int))
    assert callable(interpolation_func)
    assert isinstance(standardize, bool)
    assert (isinstance(motion_confounds, list)
        or isinstance(motion_confounds, tuple))
    assert isinstance(motion_confounds[0], str)
    # multithreading option
    assert isinstance(ncore, int)
    assert isinstance(nthread, int)
    progress_bar.update(1)
    ###########################################################################
    # load bids layout

    if from_root:
        progress_bar.set_description("loading bids dataset..".ljust(50))
        layout = BIDSLayout(root, derivatives=True)
    else:
        progress_bar.set_description("loading layout..".ljust(50))

    subjects = layout.get_subjects()
    n_subject, n_session, n_run, _0, _1 = _get_metainfo(layout)
    progress_bar.update(1)
    ###########################################################################
    # make voxel mask

    progress_bar.set_description("making custom voxel mask..".ljust(50))
    if mask_path is None:
        # if mask_path is not provided, find mask files in DEFAULT_MASK_DIR
        mask_path = Path(
            layout.derivatives["fMRIPrep"].root) / config.DEFAULT_MASK_DIR
      
    voxel_mask, masker = _custom_masking(
        mask_path, threshold, zoom,
        smoothing_fwhm, interpolation_func, standardize
    )
    progress_bar.update(1)
    ###########################################################################
    # setting parameter

    progress_bar.set_description(
        "image preprocessing - parameter setting..".ljust(50))
    params = []
    for subject in subjects:
        # get a list of nii file paths of fMRI images spread in BIDS layout
        nii_layout = layout.derivatives["fMRIPrep"].get(
            subject=subject, return_type="file", suffix="bold",
            extension="nii.gz") 
        # get a list of tsv file paths of regressors spread in BIDS layout
        # e.g. tsv file with motion confound parameters. 
        reg_layout = layout.derivatives["fMRIPrep"].get(
            subject=subject, return_type="file", suffix="regressors",
            extension="tsv")

        param = [nii_layout, reg_layout, motion_confounds,
                 masker, voxel_mask, subject]
        params.append(param)

    assert len(params) == n_subject, (
        "The length of params list and number of subjects are not validated."
    )
    progress_bar.update(1)
    ###########################################################################
    # create path for data

    progress_bar.set_description(
        "image preprocessing - making path..".ljust(50))
    if save_path is None:
        sp = Path(layout.derivatives["fMRIPrep"].root) / config.DEFAULT_SAVE_DIR
        if not sp.exists():
            sp.mkdir()
    else:
        sp = Path(save_path)
    
    nib.save(voxel_mask, sp / config.DEFAULT_VOXEL_MASK_FILENAME)
    progress_bar.update(1)
    ###########################################################################
    # Image preprocessing using mutli-processing and threading

    progress_bar.set_description(
        "image preprocessing - fMRI data..".ljust(50))
    # "chunk_size" is the number of threads.
    # In generalm this number improves performance as it grows,
    # but we recommend less than 4 because it consumes more memory.
    # TODO: We can specify only the number of threads at this time,
    #       but we must specify only the number of cores or both later.
    chunk_size = 4 if nthread > 4 else nthread
    params_chunks = [params[i:i + chunk_size]
                        for i in range(0, len(params), chunk_size)]
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
            future_result = {
                executor.submit(
                    _image_preprocess_multithreading, param, n_run): \
                        param for param in params_chunk
            }

            for future in as_completed(future_result):
                data, subject = future.result()
                np.save(
                    sp / f"{config.DEFAULT_FEATURE_PREFIX}_{subject}.npy",
                    data)
                X.append(data)

            progress_bar.set_description(
                f"image preprocessing - fMRI data.. {i+1} / {task_size} done.."
                .ljust(50))

    X = np.array(X)
    progress_bar.update(1)

    progress_bar.set_description("bids preprocessing done!".ljust(50))
    return X, voxel_mask, layout, layout.derivatives["fMRIPrep"].root
