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


def _get_metainfo(layout):
    """Get meta information of fMRI experiment.
    
    Args:
        layout (bids.BIDSLayout): BIDSLayout by bids package.
        
    Returns:
        tuple[int, int, int, int, float]:
        - **n_subject** (*int*) - the number of subjects
        - **n_session** (*int*): the number of sessions
        - **n_run** (*int*) - the number of runs
        - **n_scans** (*int*) - the time length in a single run. 
        - **t_r** (*float*) - time resolution (second) of scanning
    """
    
    n_subject = len(layout.get_subjects())
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())

    image_sample = nib.load(
        layout.derivatives["fMRIPrep"].get(
            return_type="file",
            suffix="bold",
            extension="nii.gz")[0])
    n_scans = image_sample.shape[-1]
    t_r = layout.get_tr()
    
    return (n_subject, n_session, n_run, n_scans, t_r)


def _make_single_time_mask(condition, df_events, time_length, t_r, 
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

    mask = [condition(row) for _, row in df_events.iterrows()]
    time_mask = np.zeros(time_length)

    for do_use, onset, duration in zip(mask, onsets, durations):
        if do_use:
            time_mask[int(onset / t_r): int((onset + duration) / t_r)] = 1

    return time_mask


def _make_total_time_mask(condition, df_events_list, time_length, t_r,
                          n_session, use_duration=False):
    """
    Get binary masked data indicating time points in use
    """
    """
    Binary mask indicating valid time points will be obtained by applying user-defined function "condition"
    "condition" function will censor each trial to decide whether include it or not.
    
    If use_duration == True then "duration" column data will be considered as a valid duration for selected trials,
    else the gap between consecutive trials will be used instead.

    Arguments:
        condition (function(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "events.tsv" files
        time_length (int): the length of target BOLD signal 
        t_r (float): time resolution
        use_duration (boolean): if True, use "duration" column for masking, 
                      else use the gap between consecutive onsets as duration
    Return:
        time_mask (numpy.array): binary array with shape: subject # x run # x time_length
    """

    time_mask = []
    for name0, group0 in pd.concat(df_events_list).groupby(["subjID"]):
        time_mask_subject = []
        if n_session:
            for name1, group1 in group0.groupby(["session"]):
                for name2, group2 in group1.groupby(["run"]):
                    time_mask_subject.append(_make_single_time_mask(
                        condition, group2, time_length, t_r, use_duration))
        else:
            for name1, group1 in group0.groupby(["run"]):
                time_mask_subject.append(_make_single_time_mask(
                    condition, group1, time_length, t_r, use_duration))

        time_mask.append(time_mask_subject)

    time_mask = np.array(time_mask)
    
    return time_mask

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
    df_events = df_events.sort_values(by="onset")

    def _add(row, info):
        row["subjID"] = info["subject"]
        row["run"] = info["run"]
        if "session" in info.keys():
            row["session"] = info["session"]  # if applicable
        return row

    for _, row in df_events.iterrows():
        new_df.append(_add(row, event_infos))

    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()

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
    df_events = df_events.sort_values(by="onset")

    for _, row in df_events.iterrows():
        new_df.append(preprocess(row))

    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()

    return new_df


def _process_behavior_dataframes(preprocess, df_events_list, event_infos_list):
    """
    Process columns in events file
    """
    """
    Arguments:
        preprocess (function(pandas.Series, dict)-> pandas.Series)):
            a user-defined function for modifying each row of behavioral data. 
            - f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data check
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "events.tsv" files
        event_infos_list (list(dict)): a list of dictionary containing information for each file. (ID number for subject, session, run)
    
    Return:
        df_events_list (list(pandas.DataFrame)): a list of preprocessed dataframe with required info for computaional modeling, and BOLDifying.
    """

    # add event info to each dataframe row
    df_events_list = [
        _add_event_info(df_events, event_infos)
        for df_events, event_infos in zip(df_events_list, event_infos_list)
    ]

    # modify trial data by user-defined function "preprocess"
    df_events_list = [
        _preprocess_event(
            preprocess, df_events
        ) for df_events, event_infos in zip(df_events_list, event_infos_list)
    ]
    return df_events_list


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
    return \
        individual_params[individual_params["subjID"] == int(subject_id)]


def _get_individual_params(individual_params, dm_model, condition_for_modeling,
                           df_events_list_, **kwargs):
    """
    Get individual parameter values of the model, either obtained from fitting hierarchical bayesian model supported by hBayesDM package or
    provided by user through the "individual_params" argument.
    """
    """
    Arguments:
        individual_params (str or Path or pandas.DataFrame) : pandas dataframe with params_name columns and corresponding values for each subject. if not provided, it will be obtained by fitting hBayesDM model
        dm_model (str or hbayesdm.model) : computational model by hBayesDM package. should be provided as the name of the model (e.g. "ra_prospect") or a model object.
        condition_for_modeling (function(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data which will be used for fitting computational model.
            - f(single_row_data_frame) -> True or False
        df_events_list_ (list(pandas.DataFrame)): a list of dataframe retrieved from "events.tsv" files and preprocessed in the previous stage.
    
    Returns:
        individual_params (pandas.DataFrame): the dataframe containing parameter values for each subject
        dm_model (None or hbayesdm): the fitted model if fitting happens, otherwise, None
        
    """
    
    if individual_params is None:
        # the case user does not provide individual model parameter values
        # obtain parameter values using hBayesDM package

        assert dm_model is not None, (
            "if df_events is None, must be assigned to dm_model.")

        df_events_list = [df_events[[condition_for_modeling(row) \
                            for _, row in df_events.iterrows()]] \
                                for df_events in df_events_list_]
        
        if type(dm_model) == str:
            dm_model = getattr(
                hbayesdm.models, dm_model)(
                    data=pd.concat(df_events_list),
                    **kwargs)

        individual_params = pd.DataFrame(dm_model.all_ind_pars)
        individual_params.index.name = "subjID"
        individual_params = individual_params.reset_index()
        individual_params["subjID"] = individual_params["subjID"].astype(int)
    else:
        if type(individual_params) == str\
            or type(individual_params) == type(Path()):

            individual_params = pd.read_table(individual_params)
            individual_params["subjID"] = \
                individual_params["subjID"].astype(int)
        else:
            assert type(individual_params) == pd.DataFrame
        dm_model = None
        
    return individual_params, dm_model


def _add_latent_process_single_eventdata(modulation, condition,
                                         df_events, param_dict):
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
        if condition is not None and condition(row):
            new_df.append(modulation(row, param_dict))

    new_df = pd.concat(
        new_df, axis=1, keys=[s.name for s in new_df]
    ).transpose()

    return new_df


def _add_latent_process_as_modulation(individual_params, modulation, condition,
                                      df_events_list, event_infos_list):
    """
    Calculate latent process using user-defined function "modulation", and add it to dataframe as a "modulation" column.
    """
    """
    Arguments:
        individual_params (pandas.DataFrame): the dataframe containing parameter values for each subject
        modulation (function(pandas.Series, dict)-> Series): a user-defined function for calculating latent process (modulation). 
            - f(single_row_data_frame, model_parameter_dict) -> single_row_data_frame_with_latent_state 
        condition (function(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "events.tsv" files and preprocessed in the previous stage.
        event_infos_list (list(dict)): a list of dictionary containing information for each file. (ID number for subject, session, run)
    
    Returns:
        df_events_ready (pandas.DataFrame): a integrated dataframe with "modulation" which is ready for BOLDification

    """
    df_events_list = [
            _add_latent_process_single_eventdata(
                modulation, condition, df_events,
                _get_individual_param_dict(
                    event_infos["subject"], individual_params)
            ) for df_events, event_infos in \
                 zip(df_events_list, event_infos_list)]
    
    df_events_ready =  pd.concat(df_events_list)
    return df_events_ready


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


def _convert_event_to_boldlike_signal(df_events, t_r, n_scans, is_session,
                                      hrf_model="glover",
                                      normalizer="minmax"):
    
    """
    BOLDify the preprocessed behavioral (event) data 
    by converting behavior data to weighted impulse seqeunces and convolve them with hemodynamic response function. 
    """
    """
    This is done by utilizing nilearn.glm.first_level.hemodynamic_models.compute_regressor,
    by providing "onset","duration", and "modulation" values.
    the final matrix will be shaped as the number of subject * of run * of n_scan
    "n_scan" means the number of time points in fMRI data
    if there is multiple session, still there would be no dimension indicating sessions info, 
    but runs will be arranged as grouped by sessions number.
    e.g. (subj-01, sess-01, run-01,:)
         (subj-01, sess-01, run-02,:)
                    ...
         (subj-01, sess-02, run-01,:)
         (subj-01, sess-02, run-02,:)
                    ...
    This order should match with preprocessed fMRI image data.
    
    Arguments:
        df_events (DataFrame): preprocessed dataframe which is ready to BOLDify.
        t_r (float): time resolution (second).
        hrf_model(str): name for hemodynamic model.
        normalizer(str): method name for normalizing output BOLD-like signal. "minmax" or "standard" (equal z_score).
        
    Return:
        boldified_signals (numpy.array): BOLD-like signal.
    """
    # The reason for adding t_r / 2 is to provide sufficient convolution operation.
    frame_times = t_r * (np.arange(n_scans) + t_r / 2)

    signals = []
    for name0, group0 in df_events.groupby(["subjID"]):
        signal_subject = []
        if is_session:
            for name1, group1 in group0.groupby(["session"]):
                for name2, group2 in group1.groupby(["run"]):
                    modulation_ = group2[
                        ["onset", "duration", "modulation"]].to_numpy(
                            dtype=float).T
                    signal, _ = _boldify(
                        modulation_, hrf_model, frame_times)
                    signal_subject.append(signal)
        else:
            for name1, group1 in group0.groupby(["run"]):
                modulation_ = group1[
                ["onset", "duration", "modulation"]].to_numpy(dtype=float).T
                signal, _ = _boldify(modulation_, hrf_model, frame_times)
                signal_subject.append(signal)

        signal_subject = np.array(signal_subject)
        reshape_target = signal_subject.shape

        # method for normalizing signal
        if normalizer == "standard":
            # standard normalization by calculating zscore
            normalized_signal = zscore(signal_subject.flatten(), axis=None)
        else:
            # default is using minmax
            normalized_signal = minmax_scale(
                signal_subject.flatten(),
                feature_range=(-1, 1), axis=0)

        normalized_signal = normalized_signal.reshape(reshape_target)
        signals.append(normalized_signal)
    signals = np.array(signals)
    
    return signals
