#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

"""
Helper functions for preprocessing behavior data
"""

from pathlib import Path
import hbayesdm.models
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level.hemodynamic_models import compute_regressor
from scipy.stats import zscore
from sklearn.preprocessing import minmax_scale

from ..utils import config # configuration for default names used in the package


    
def _process_indiv_params(individual_params):
    if type(individual_params) == str\
        or type(individual_params) == type(Path()):
        try:
            individual_params = pd.read_table(individual_params)
            print("INFO: individual parameters are loaded")
        except:
            individual_params = None
        
        return individual_params
    elif type(individual_params) == pd.DataFrame:
        return individual_params
    else:
        return None
    
def _add_event_info(df_events, event_infos):
    """
    Add subject, run, session info to dataframe of events of single "run" 
    """
    """
    Arguments:
        df_events (pandas.Dataframe): a dataframe retrieved from "events.tsv" file
        event_infos (dict): a dictionary containing "subject", "run", (and "session" if applicable).

    Return:
        new_df (pandas.Dataframe): a dataframe with event info
    """

    new_df = []

    def _add(i, row, info):
        row["trial"] = i 
        row["subjID"] = info["subject"]
        row["run"] = info["run"]
        if "session" in info.keys():
            row["session"] = info["session"]  # if applicable
        return row

    for i, row in df_events.iterrows():
        new_df.append(_add(i, row, event_infos))
    
    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()
    
    #new_df pd.concat(new_df)
    
    return new_df

def _preprocess_event(preprocess, df_events):
    """
    Preprocess dataframe of events of single "run" 
    """
    """
    Arguments:
        preprocess (function(pandas.Series, dict)-> pandas.Series)):
            a user-defined function for modifying each row of behavioral data.
            - f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data check
        df_events (pandas.Dataframe): a dataframe retrieved from "events.tsv" file

    Return:
        new_df (pandas.Dataframe): a dataframe with preprocessed rows
    """

    new_df = []

    for _, row in df_events.iterrows():
        new_df.append(preprocess(row))

    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()
    #new_df = new_df.sort_values(by="onset")
    return new_df

def _make_single_time_mask(condition, df_events, time_length, t_r, 
                           preprocess=None,
                           use_duration=False):
    """
    Get binary masked data indicating time points in use
    """
    """
    Arguments:
        condition (function(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events (pandas.Dataframe): a dataframe retrieved from "events.tsv" file
        time_length (int): the length of target BOLD signal 
        t_r (float): time resolution
        use_duration (boolean): if True, use "duration" column for masking, 
                      else use the gap between consecutive onsets as duration
    Return:
        time_mask (numpy.array): binary array with shape: time_length
    """
    df_events = df_events.sort_values(by="onset")
    onsets = df_events["onset"].to_numpy()
    if use_duration:
        durations = df_events["duration"].to_numpy()
    else:
        durations = np.array(
            list(df_events["onset"][1:]) + [time_length * t_r]) - onsets
    if callable(preprocess):
        mask = [condition(preprocess(row)) for _, row in df_events.iterrows()]
    else:
        mask = [condition(row) for _, row in df_events.iterrows()]
    time_mask = np.zeros(time_length)

    for do_use, onset, duration in zip(mask, onsets, durations):
        if do_use:
            time_mask[int(onset / t_r): int((onset + duration) / t_r)] = 1

    return time_mask

def _get_individual_param_dict(subject_id, individual_params):
    """
    Get individual parameter dictionary
    so the value can be referred by its name (type:str)
    """
    """
    Arguments:
        subject_id (int or str): subject ID number
        individual_params (pandas.DataFrame): pandas dataframe with individual parameter values where each row number matches with subject ID.

    Return:
        ind_pars (dict): individual parameter value. dictionary{parameter_name:value}

    """
    idp = individual_params[individual_params["subjID"] == subject_id]
    if len(idp) == 0:
        idp = individual_params[individual_params["subjID"] == int(subject_id)]
    if len(idp) == 0:
        return None
    return idp

def _add_latent_process_single_eventdata(modulation, condition,
                                         df_events, param_dict, preprocess=None):
    """
    Add latent state value to for each row of dataframe of single "run"
    """
    """
    Argumnets:
        modulation (function(pandas.Series, dict)-> Series): a user-defined function for calculating latent process (modulation). 
            - f(single_row_data_frame, model_parameter_dict) -> single_row_data_frame_with_latent_state 
        condition (function(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events (pandas.DataFrame): a dataframe retrieved from "events.tsv" file and preprocessed in the previous stage.
        param_dict (dict): a dictionary containing model parameter value

    Return:
        new_df (pandas.DataFrame): a dataframe with latent state value ("modulation")
    """
    new_df = []
    df_events = df_events.sort_values(by="onset")
    for _, row in df_events.iterrows():
        if callable(preprocess):
            row = preprocess(row)
        if condition is not None and condition(row):
            new_df.append(modulation(preprocess(row), param_dict)[['onset','duration','modulation']])
    new_df = pd.concat(
        new_df, axis=1, keys=[s.name for s in new_df]
    ).transpose()

    return new_df

def _boldify(modulation_, hrf_model, frame_times):
    """
    BOLDify event data.
    by converting behavior data to weighted impulse seqeunce and convolve it with hemodynamic response function. 
    """
    """
    Arguments:
        modulation_ (numpy.array): a array with onset, duration, and modulation values. shape : 3 x time point #
        hrf_model (str): name for hemodynamic model
        frame_times (numpy.array or list): frame array indicating time slot for BOLD-like signals
        
    Return:
        boldified_signals (numpy.array): BOLD-like signal. 
        
    """

    boldified_signals, name = compute_regressor(
        exp_condition=modulation_.astype(float),
        hrf_model=hrf_model,
        frame_times=frame_times)

    return boldified_signals, name
