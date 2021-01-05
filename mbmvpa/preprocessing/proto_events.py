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
              root=None,
              layout=None,
              save_path=None,
              preprocess=None,
              condition=None,
              modulation=None,
              dm_model=None,
              condition_for_modeling=None,
              individual_params_custom=None,
              hrf_model="glover",
              normalizer="minmax",
              df_events_custom=None,
              use_duration=False,
              scale=(-1, 1)):

        # setting path informations and loading layout
        if root is None:
            assert layout is not None
            from_root = False
            self.root = layout.root
            self.layout = layout
        else:
            self.root = root
            self.layout = BIDSLayout(root, derivatives=True)
            
        assert (save_path is None
            or isinstance(save_path, str)
            or isinstance(save_path, Path))
        
        if save_path is None:
            sp = Path(
                layout.derivatives["fMRIPrep"].root) / config.DEFAULT_SAVE_DIR
        else:
            sp = Path(save_path)

        if not sp.exists():
            sp.mkdir()

        self.save_path = sp

        # setting meta-info
        self.n_subject, self.n_session, self.n_run, self.n_scans, self.t_r = _get_metainfo(self.layout)
        
        # setting user-defined functions
        self.preprocess = preprocess
        self.condition = conditon
        self.modulation = modulation
        
        # setting model fitting specification
        self.dm_model = dm_model
        self.condtion_for_modeling = condition_for_modeling
        self.individual_params = individual_params_custom
        
        # setting BOLD-like signal generating specification
        self.hrf_model = hrf_model
        self.normalizer = normalizer
        self.scale = scale
        
        # setting attribute holding data frames for event data
        self._df_events_ready = df_events_custom
        
        # setting working space
        # this will aggregate all events file path in sorted way
        events = layout.get(suffix="events", extension="tsv")
        # collecting dataframe from event files spread in BIDS layout
        self._df_events_list = [event.get_df() for event in events]
        # event_info contains ID number for subject, session, run
        self._event_infos_list = [event.get_entities() for event in events]
        
        #TODO pring basic meta info of BIDS layout
        
        