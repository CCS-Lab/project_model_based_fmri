#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong, Cheoljun cho
@contact: mybirth0407@gmail.com
          cjfwndnsl@gmail.com
@last modification: 2020.11.16

This code is for preprocessing behavior data ("events.tsv") to convert them to BOLD-like signals.

"""

import logging
import time
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

import hbayesdm.models
import time

import logging

from ..utils import config # configuration for default names used in the package
logging.basicConfig(level=logging.INFO)


def events_preprocess(# path info
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
                      individual_params=None,
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
    """
    This function is for preprocessing behavior data ("events.tsv") to convert them to BOLD-like signals.
    The BOLD-like signals will be used for a target(y) in MVPA.
    Also, it will produce time masks which are binary arrays with the same size as the time dimension of the data
    to indicate which time point of data will be included in MVPA.
    
    Note :
        The default setting is calculating latent process (or 'modulation') 
        by using hierarchical Bayesian modeling by running "hBayesDM" package.
        
        User can optionally skip the steps in this process in the following possible scenarios
        1) User can provide precalculated behavioral data through "df_events_custom" argument. 
           In this case, it will skip both fitting model and extracting latent process. 
        2) User can provide precalculated individual model parameter values through "individual_params" argument.
           In this case, it will only skip model fitting part.
           
    Arguments:
        root (str or Path): the root directory of BIDS layout
        layout (nibabel.BIDSLayout): BIDSLayout by bids package. if not provided, it will be obtained from root path.
        save_path (str or Path): a path for the directory to save outputs (y, time_mask) and intermediate data (individual_params, df_events). if not provided, "BIDS root/derivatives/data" will be set as default path      
        preprocess (func(pandas.Series, dict)-> pandas.Series)): a user-defined function for modifying each row of behavioral data. 
            - f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data check
        condition (func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        modulation (func(pandas.Series, dict)-> Series): a user-defined function for calculating latent process (modulation). 
            - f(single_row_data_frame, model_parameter_dict) -> single_row_data_frame_with_latent_state 
        condition_for_modeling (None or func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data which will be used for fitting computational model.
            - None : "condition" function will be used.
            - f(single_row_data_frame) -> True or False
        dm_model (str or hbayesdm.model) : computational model by hBayesDM package. should be provided as the name of the model (e.g. 'ra_prospect') or a model object.
        individual_params (str or Path or pandas.DataFrame) : pandas dataframe with params_name columns and corresponding values for each subject. if not provided, it will be obtained by fitting hBayesDM model
        hrf_model (str): the name for hemodynamic response function, which will be convoluted with event data to make BOLD-like signal
            the below notes are retrieved from the code of "nilearn.glm.first_level.hemodynamic_models.compute_regressor"
            (https://github.com/nilearn/nilearn/blob/master/nilearn/glm/first_level/hemodynamic_models.py)
            
            The different hemodynamic models can be understood as follows:
                 - 'spm': this is the hrf model used in SPM
                 - 'spm + derivative': SPM model plus its time derivative (2 regressors)
                 - 'spm + time + dispersion': idem, plus dispersion derivative
                                            (3 regressors)
                 - 'glover': this one corresponds to the Glover hrf
                 - 'glover + derivative': the Glover hrf + time derivative (2 regressors)
                 - 'glover + derivative + dispersion': idem + dispersion derivative
                                                    (3 regressors)
            
        normalizer (str): a name for normalization method, which will normalize BOLDified signal. 'minimax' or 'standard' 
            - 'minmax': rescale value by putting minimum value and maximum value for each subject to be given lower bound and upper bound respectively
            - 'standard': rescale value by calculating subject-wise z_score
        use_duration (boolean) : if True use 'duration' column to make time mask, if False regard gap between consecuting trials' onset values as duration
        scale (tuple(float, float)) : lower bound and upper bound for minmax scaling. will be ignored if 'standard' normalization is selected. default is -1 to 1.

    Returns:
        dm_model (hbayesdm.model): hBayesDM model.
        df_events (pandas.DataFrame): integrated event DataFrame (preprocessed if not provided) with 'onset','duration','modulation'
        signals (numpy.array): BOLD-like signals with shape: subject # x (session # x run #) x time length of scan x voxel #
        time_mask (numpy.array): a  binary mask indicating valid time point with shape: subject # x (session # x run #) x time length of scan
    """

    progress_bar = tqdm(total=6)
    s = time.time()

    ###########################################################################
    # load data from bids layout

    if layout is None:
        progress_bar.set_description("loading bids dataset..".ljust(50))
        layout = BIDSLayout(root, derivatives=True)
    else:
        progress_bar.set_description("loading layout..".ljust(50))
    
    # get meta info
    n_subject, n_session, n_run, n_scans, t_r = _get_metainfo(layout)
    
    # this will aggregate all events file path in sorted way
    events = layout.get(suffix="events", extension="tsv")
    # collecting dataframe from event files spread in BIDS layout
    df_events_list = [event.get_df() for event in events]
    # event_info contains ID number for subject, session, run
    event_infos_list = [event.get_entities() for event in events]
    
    progress_bar.update(1)

    ###########################################################################
    # designate saving path

    if save_path is None:
        sp = Path(
            layout.derivatives["fMRIPrep"].root) / config.DEFAULT_SAVE_DIR
    else:
        sp = Path(save_path)

    if not sp.exists():
        sp.mkdir()

    ###########################################################################
    # process columns in events file

    progress_bar.set_description("processing event file columns..".ljust(50))

    df_events_list = _process_behavior_dataframes(
        preprocess,df_events_list,event_infos_list)
    
    progress_bar.update(1)

    ###########################################################################
    # get time masks

    progress_bar.set_description("calculating time masks..".ljust(50))

    time_mask = _make_total_time_mask(
        condition, df_events_list, n_scans, t_r, n_session, use_duration)

    np.save(sp / config.DEFAULT_TIME_MASK_FILENAME, time_mask) 

    progress_bar.update(1)

    ###########################################################################
    # Get dataframe with 'subjID','run','duration','onset','duration' and 'modulation' which are required fields for making BOLD-like signal
    # if user provided the "df_events" with those fields, this part will be skipped

    # the case user does not provide precalculated behavioral data.
    if df_events_custom is None:
        assert modulation is not None, (
            "if df_events is None, must be assigned to latent_function")
        
        if condition_for_modeling is None:
            condition_for_modeling = condition
        
        progress_bar.set_description(
            "hbayesdm doing (model: %s)..".ljust(50) % dm_model)
        # get individual parameter values in computational model which will be used to calculate the latent process('modulation').
        individual_params, dm_model = _get_individual_params(
            individual_params,dm_model,
            condition_for_modeling,
            df_events_list,
            **kwargs)
        # if calculate individual params.
        if dm_model is not None:
            individual_params.to_csv(
                sp / config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME,
                sep="\t")
        
        progress_bar.update(1)
        progress_bar.set_description("calculating modulation..".ljust(50))
        # the 'modulation' values are obtained by applying user-defined function "modulation" with model parameter values
        df_events_ready = _add_latent_process_as_modulation(
            individual_params, modulation,
            condition,
            df_events_list,
            event_infos_list)
        
        progress_bar.update(1)
    else:
        # sanity check. the user provided dataframe should contain following data.
        # else, raise error.
        assert ("modulation" in df_events_custom.columns
            and "subjID" in df_events_custom.columns
            and "run" in df_events_custom.columns
            and "onset" in df_events_custom.columns
            and "duration" in df_events_custom.columns
            and "modulation" in df_events_custom.columns),\
        ("missing column in behavior data")
        
        df_events_ready = df_events_custom
        progress_bar.update(2)
        
    ###########################################################################
    # Get boldified signals.

    progress_bar.set_description("modulation signal making..".ljust(50))
    signals = _convert_event_to_boldlike_signal(
        df_events_ready, t_r, hrf_model, normalizer)
    progress_bar.update(1)
    
    np.save(sp / config.DEFAULT_MODULATION_FILENAME, signals)
    progress_bar.update(1)

    ###########################################################################
    # elapsed time check

    progress_bar.set_description("events preproecssing done!".ljust(50))

    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")

    return dm_model, df_events, signals, time_mask, layout
