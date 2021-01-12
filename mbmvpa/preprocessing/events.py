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
from .events_utils import _get_metainfo, _add_event_info, _preprocess_event
from .events_utils import _process_indiv_params,_add_latent_process_single_eventdata, _get_individual_param_dict
from .events_utils import get_time_mask, convert_event_to_boldlike_signal
from bids import BIDSLayout
from tqdm import tqdm

from ..utils import config # configuration for default names used in the package


def events_preprocess(# path informations
                      root=None,
                      layout=None,
                      save_path=None,
                      # user-defined functions
                      preprocess=lambda x: x,
                      condition=lambda _: True,
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
                      scale=(-1, 1),
                      # hBayesDM fitting parameters
                      **kwargs,
                      ):
    
    generator = LatentProcessGenerator(root=root,
                                      layout=layout,
                                      save_path=save_path,
                                      preprocess=preprocess,
                                      condition=condition,
                                      modulation=modulation,
                                      dm_model=dm_model,
                                      condition_for_modeling=condition_for_modeling,
                                      individual_params_custom=individual_params_custom,
                                      hrf_model=hrf_model,
                                      normalizer=normalizer,
                                      df_events_custom=df_events_custom,
                                      use_duration=use_duration,
                                      scale=scale)
    
    boldsignals, time_mask = generator.run()
    
    return (generator._trained_dm_model, generator._df_events_ready, 
         boldsignals, time_mask, generator.layout)

class LatentProcessGenerator():
    def __init__(self, 
              root=None,
              layout=None,
              save_path=None,
              preprocess=lambda x: x,
              condition=lambda _: True,
              modulation=None,
              dm_model=None,
              condition_for_modeling=None,
              individual_params_custom=None,
              hrf_model="glover",
              normalizer="standard-minimax",
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
                self.layout.derivatives["fMRIPrep"].root) / config.DEFAULT_SAVE_DIR
        else:
            sp = Path(save_path)

        if not sp.exists():
            sp.mkdir()

        self.save_path = sp

        # setting meta-info
        self.n_subject, self.n_session, self.n_run, self.n_scans, self.t_r = _get_metainfo(self.layout)
        
        # setting user-defined functions
        self.preprocess = preprocess
        self.condition = condition
        self.modulation = modulation
        
        # setting model fitting specification
        self.dm_model = dm_model
        self.condition_for_modeling = condition_for_modeling
        self.individual_params = _process_indiv_params(individual_params_custom)
        self._idividual_params_provided = self.individual_params is not None

        # setting BOLD-like signal generating specification
        self.hrf_model = hrf_model
        self.normalizer = normalizer
        self.scale = scale
        
        self.use_duration = use_duration

        # setting attribute holding data frames for event data
        if df_events_custom is not None:
            sanity_check = ("modulation" in df_events_custom.columns
                            and "subjID" in df_events_custom.columns
                            and "run" in df_events_custom.columns
                            and "onset" in df_events_custom.columns
                            and "duration" in df_events_custom.columns)
            if sanity_check:
                self._df_events_ready = df_events_custom
            else:
                self._df_events_ready = None
        else:
            self._df_events_ready = None
            
        self._df_events_ready_provided = self._df_events_ready is not None

        # initial working space
        self._df_events_list = None
        self._event_infos_list = None
        self.time_mask = None
        self._trained_dm_model = None
        #TODO pring basic meta info of BIDS layout

    def clean(self):
        self._df_events_list = None
        self._event_infos_list = None
        self.time_mask = None
        self._trained_dm_model = None
        self._df_events_ready = None
        self.individual_params 

        if not self._idividual_params_provided:
            self.individual_params = None
        if not self._df_events_ready_provided:
            self._df_events_ready = None

    def run(self, clean=False,  **kwargs):

        # TODO add progress bar
        
        if clean:
            self.clean()

        if self._df_events_ready is None:
            if self._df_events_list is None or self._event_infos_list:
                self.init_df_events_from_bids()

            if self.individual_params is None:
                self.set_computational_model(**kwargs)

            self.set_df_events_ready()

        self.set_time_mask()
        boldsignals = self.generate_boldlike_signal()
        np.save(self.save_path  / config.DEFAULT_MODULATION_FILENAME, boldsignals)
        
        return boldsignals, self.time_mask


    def init_df_events_from_bids(self, preprocess=None):

        if preprocess is None:
            preprocess = self.preprocess

        # this will aggregate all events file path in sorted way
        events = self.layout.get(suffix="events", extension="tsv")
        # collecting dataframe from event files spread in BIDS layout
        self._df_events_list = [event.get_df() for event in events]
        # event_info contains ID number for subject, session, run
        self._event_infos_list = [event.get_entities() for event in events]
        # add event info to each dataframe row
        self._df_events_list = [
            _add_event_info(df_events, event_infos)
            for df_events, event_infos in zip(self._df_events_list, self._event_infos_list)
        ]

        if callable(preprocess):
            # modify trial data by user-defined function "preprocess"
            self._df_events_list = [
                _preprocess_event(
                    preprocess, df_events
                ) for df_events, event_infos in zip(self._df_events_list, self._event_infos_list)
            ]

    def set_time_mask(self, df_events=None, condition=None, use_duration=None):

        if df_events is None:
            if self._df_events_ready is not None:
                df_events = self._df_events_ready
            elif self._df_events_list is not None:
                df_events = pd.concat(self._df_events_list)
            else:
                assert False
        if use_duration is None:
            use_duration = self.use_duration
        if condition is None:
            condition = self.condition

        self.time_mask = get_time_mask(df_events, self.n_scans, self.t_r, self.n_session,
                                       condition, use_duration)
        np.save(self.save_path / config.DEFAULT_TIME_MASK_FILENAME, self.time_mask) 


    def set_computational_model(self, df_events=None, individual_params=None, dm_model=None, condition=None, **kwargs):

        if df_events is None:
            assert self._df_events_list is not None
            df_events = pd.concat(self._df_events_list)
        individual_params = _process_indiv_params(individual_params)
        if individual_params is None:
            individual_params = self.individual_params
        if dm_model is None:
            dm_model = self.dm_model
        if condition is None:
            if self.condition_for_modeling is None:
                condition = self.condition
            else :
                condition = self.condition_for_modeling

        if individual_params is None :
            # the case user does not provide individual model parameter values
            # obtain parameter values using hBayesDM package

            assert dm_model is not None, (
                "if df_events is None, must be assigned to dm_model.")

            df_events_list = [df_events[[condition(row) \
                                for _, row in df_events.iterrows()]] \
                                    for df_events in self._df_events_list]

            if type(dm_model) == str:
                model = getattr(
                    hbayesdm.models, dm_model)(
                        data=pd.concat(df_events_list),
                        **kwargs)

            individual_params = pd.DataFrame(model.all_ind_pars)
            individual_params.index.name = "subjID"
            individual_params = individual_params.reset_index()
            individual_params["subjID"] = individual_params["subjID"].astype(int)
            individual_params.to_csv(
                self.save_path / config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME,
                sep="\t", index=False)
            self._trained_dm_model = model

        self.individual_params = individual_params

    def set_df_events_ready(self, individual_params=None, modulation=None, condition=None):

        if individual_params is None:
            individual_params = self.individual_params
        if modulation is None:
            modulation = self.modulation
        if condition is None:
            condition = self.condition

        assert self._df_events_list is not None, (
                "Please run init_df_events_from_bids first")
        assert self._event_infos_list is not None, (
                "Please run init_df_events_from_bids first")
        assert individual_params is not None, (
                "Please run set_computational_model first")
        assert callable(modulation), (
                "Please provide a valid user-defined modulation function")

        df_events_list = [
            _add_latent_process_single_eventdata(
                modulation, condition, df_events,
                _get_individual_param_dict(
                    event_infos["subject"], individual_params)
            ) for df_events, event_infos in \
                 zip(self._df_events_list, self._event_infos_list)]

        self._df_events_ready =  pd.concat(df_events_list)
        self._df_events_ready['modulation'] = self._df_events_ready['modulation'].astype('float')
        
    def generate_boldlike_signal(self, hrf_model=None, normalizer=None, scale=None):

        if hrf_model is None:
            hrf_model = self.hrf_model
        if normalizer is None:
            normalizer = self.normalizer
        if scale is None:
            scale = self.scale

        assert isinstance(hrf_model, str)
        assert isinstance(normalizer, str)
        assert (isinstance(scale, list)
            or isinstance(scale, tuple))

        return convert_event_to_boldlike_signal(self._df_events_ready, 
                                                self.t_r, self.n_scans, self.n_session,
                                                hrf_model=hrf_model,
                                                normalizer=normalizer,
                                                scale=scale)

