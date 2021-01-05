#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2021.01.05
## class version 

"""
It is for preprocessing behavior data ("events.tsv") to convert them to BOLD-like signals.
The result BOLD-like signals will be used for a target(y) in MVPA.
Also, it will produce time masks which are binary arrays with the same size as the time dimension of the data 
to indicate which time point of data will be included in MVPA.
The default setting is calculating latent process (or "modulation") 
by using hierarchical Bayesian modeling by running "hBayesDM" package.

User can optionally skip the steps in this process in the following possible scenarios
- User can provide **precalculated behavioral data** through *df_events_custom* argument. In this case, it will skip both fitting model and extracting latent process. 
- User can provide **precalculated individual model parameter values** through *individual_params_custom* argument.In this case, it will only skip model fitting part.
"""

from pathlib import Path

import hbayesdm.models
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale
from .event_utils import _get_metainfo, _process_behavior_dataframes, _make_total_time_mask
from .event_utils import _get_individual_params, _add_latent_process_as_modulation
from .event_utils import _convert_event_to_boldlike_signal
from bids import BIDSLayout
from tqdm import tqdm

from ..utils import config # configuration for default names used in the package


class LatentProcessGenerator():
    
    def __init__(self, 
                 # path informations
                  root=None,
                  layout=None,
                  save_path=None,
                  # user-defined functions
                  preprocess=None,
                  condition=None,
                  modulation=None,
                  # computational model specification
                  condition_for_modeling=None,
                  dm_model=None,
                  individual_params_custom=None,
                  # BOLDifying parameter
                  hrf_model="glover",
                  normalizer="minmax",
                  # Other specification
                  df_events_custom=None,
                  use_duration=False,
                  scale=(-1, 1)):
        
        # setting path informations
        from_root = True
        if root is None:
            assert layout is not None
            from_root = False
            self.root = layout.root
        else:
            self.root = root
            
        assert (save_path is None
            or isinstance(save_path, str)
            or isinstance(save_path, Path))

        self.save_path = save_path
        
        # setting user-defined functions
        self.preprocess = preprocess
        self.condition = conditon
        self.modulation = modulation
        
        # 