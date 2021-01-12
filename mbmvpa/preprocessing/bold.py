from concurrent.futures import ProcessPoolExecutor, as_completed
from .event_utils import _get_metainfo
from pathlib import Path
import numpy as np
import bids
from bids import BIDSLayout
from tqdm import tqdm
from .bold_utils import _custom_masking, _image_preprocess_multithreading
import nibabel as nib

from ..utils import config # configuration for default names used in the package


bids.config.set_option("extension_initial_dot", True)



class VoxelDataGenerator():
    
    def  __init__(self,
                 )