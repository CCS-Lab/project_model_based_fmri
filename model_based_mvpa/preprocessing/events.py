#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong, Cheoljun cho
@contact: mybirth0407@gmail.com
          cjfwndnsl@gmail.com
@last modification: 2020.11.16
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

from bids import BIDSLayout
from tqdm import tqdm

import hbayesdm.models
import time

import logging

from ..utils import config

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
                      dm_model=None,
                      all_individual_params_path=None,
                      # BOLDifying parameter
                      hrf_model="glover",
                      normalizer="minmax",
                      # Other specification
                      df_events_custom=None,
                      use_duration=False,
                      save=True,
                      scale=(-1, 1),
                      # hBayesDM fitting parameters
                      **kwargs,
                      ):
    """
    Preprocessing event data to get BOLD-like signal and time mask for indicating valid range of data.
    User can provide precalculated behaviral data through "df_events" argument,
    which is the DataFrame with subjID, run, onset, duration, and modulation. (also session if applicable)
    if not, it will calculate latent process by using hierarchical Bayesian modeling using "hBayesDM" package.
    User also can provide precalculated individual model parameter values, through "all_individual_params" argument.

    The time mask will be used for selecting data points included in model fitting
    The BOLD-like signal will be used for a target(y) in model fitting

    Arguments:
        root (str or Path): root directory of BIDS layout
        layout (BIDSLayout): BIDSLayout by bids package. if not provided, it will be obtained using root info.
        save_path (str or Path): path for saving output. if not provided, BIDS root/derivatives/data will be set as default path
        preprocess (func(Series, dict)-> Series)): user defined function for modifying behavioral data. f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data
        condition (func(Series)-> boolean)): user defined function for filtering behavioral data. f(single_row_data_frame) -> True or False
        modulation (func(Series, dict)-> Series): user defined function for calculating latent process. f(single_row_data_frame, model_parameter_dict) -> single_row_data_frame_with_latent_state 
        dm_model (str or hbayesdm.model) : model for hBayesDM package. should be provided as model name (e.g. 'ra_prospect') or model object.
        all_individual_params_path (str or pandas.DataFrame) : pandas dataframe with params_name columns and corresponding values for each subject. if not provided, it will be obtained by fitting hBayesDM model
        hrf_model (str): specification for hemodynamic response function, which will be convoluted with event data to make BOLD-like signal
        normalizer (str): normalization method to subject-wisely normalize BOLDified signal. "minimax" or "standard" 
                          "minmax" will rescale value by putting minimum value and maximum value for each subject to be given lower bound and upper bound respectively
                          "standard" will rescale value by calculating subject-wise z_score
        use_duration (boolean) : if True use 'duration' column info to make time mask, if False regard gap between consecuting onsets as duration
        save (boolean): indicating whether save result if True, you will save y.npy, time_mask.npy and additionaly all_individual_params.tsv.
        scale (tuple(float, float)) : lower bound and upper bound for minmax scaling. will be ignored if 'standard' normalization is selected. default is -1 to 1.

    Return
        dm_model: hBayesDM model.
        df_events: integrated event DataFrame (preprocessed if not provided) with 'onset','duration','modulation'
        signals: BOLD-like signal.
                 shape: subject # x (session # x run #) x time length of scan x voxel #
        time_mask: binary mask indicating valid time point.
                   shape: subject # x (session # x run #) x time length of scan
    """

    pbar = tqdm(total=6)
    s = time.time()

    ###########################################################################
    # load data from bids layout

    if layout is None:
        pbar.set_description("loading bids dataset..".ljust(50))
        layout = BIDSLayout(root, derivatives=True)
    else:
        pbar.set_description("loading layout..".ljust(50))

    t_r = layout.get_tr()

    # this will aggregate all events file path in sorted way
    events = layout.get(suffix="events", extension="tsv")

    subjects = layout.get_subjects()
    n_subject = len(subjects)
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())

    image_sample = nib.load(
        layout.derivatives["fMRIPrep"].get(
            return_type="file",
            suffix="bold",
            extension="nii.gz")[0]
    )
    n_scans = image_sample.shape[-1]

    # collecting dataframe from event files spread in BIDS layout
    df_events_list = [event.get_df() for event in events]

    # event_info such as id number for subject, session, run
    event_infos_list = [event.get_entities() for event in events]
    pbar.update(1)

    ###########################################################################
    # designate saving path

    if save_path is None:
        sp = Path(layout.derivatives["fMRIPrep"].root) / config.DEFAULT_SAVE_DIR
    else:
        sp = Path(save_path)

    if save and not sp.exists():
        sp.mkdir()

    ###########################################################################
    # adjust columns in events file

    pbar.set_description("adjusting event file columns..".ljust(50))

    # add event info to each dataframe row
    df_events_list = [
        _add_event_info(df_events, event_infos)
        for df_events, event_infos in zip(df_events_list, event_infos_list)
    ]

    # modify trial data by user-defined function "preprocess"
    df_events_list = [
        _preprocess_event(
            preprocess, condition, df_events
        ) for df_events, event_infos in zip(df_events_list, event_infos_list)
    ]
    pbar.update(1)

    ###########################################################################
    # get time masks
    # binary mask indicating valid time points will be obtained by applying user-defined function "condition"
    # "condition" function will censor each trial to decide whether include it or not
    # if use_duration == True then 'duration' column data will be considered as a valid duration for selected trials,
    # else the gap between consecutive trials will be used instead.

    pbar.set_description("calculating time masks..".ljust(50))

    time_mask = []
    for name0, group0 in pd.concat(df_events_list).groupby(["subjID"]):
        time_mask_subject = []
        if n_session:
            for name1, group1 in group0.groupby(["session"]):
                for name2, group2 in group1.groupby(["run"]):
                    time_mask_subject.append(_get_time_mask(
                        condition, group1, n_scans, t_r, use_duration))
        else:
            for name1, group1 in group0.groupby(["run"]):
                time_mask_subject.append(_get_time_mask(
                    condition, group1, n_scans, t_r, use_duration))

        time_mask.append(time_mask_subject)

    time_mask = np.array(time_mask)

    if save:
        np.save(sp / config.DEFAULT_TIME_MASK_FILENAME, time_mask)

    pbar.update(1)

    ###########################################################################
    # Get dataframe with 'subjID','run','duration','onset','duration' and 'modulation' which are required fields for making BOLD-like signal
    # if user provided the "df_events" with those fields, this part will be skipped
    # the fields except for 'modulation' already exist in "df_events_list",
    # so the 'modulation' values are obtained by applying user-defined function "modulation" with model parameter values
    # obtained from fitting hierarchical bayesian model supported by hBayesDM package.
    # Here, user also can provide precalculated individual model parameters in dataframe form through the "all_individual_params" argument.

    if df_events_custom is None:
        # the case user does not provide precalculated bahavioral data
        # calculate latent process using user defined latent function

        assert modulation is not None, "if df_events is None, must be assigned to latetn_function"

        if all_individual_params_path is None:
            # the case user does not provide individual model parameter values
            # obtain parameter values using hBayesDM package

            assert dm_model is not None, "if df_events is None, must be assigned to dm_model."

            pbar.set_description(
                "hbayesdm doing (model: %s)..".ljust(50) % dm_model)

            if type(dm_model) == str:
                dm_model = getattr(
                    hbayesdm.models, dm_model)(
                        data=pd.concat(df_events_list),
                        **kwargs
                    )
            
            all_individual_params = dm_model.all_ind_pars
            all_individual_params.columns[0] = "subjID"

            if save:
                all_individual_params.to_csv(
                    sp / config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME, sep='\t')
                
        else:
            assert (
                (type(all_individual_params_path) == str)
                or (type(all_individual_params_path) == Path())), (
                "")

            all_individual_params = pd.read_csv(
                all_individual_params_path, sep='\t')
            s = len(str(all_individual_params['subjID'].max()))
            all_individual_params['subjID'] =\
                all_individual_params['subjID'].apply(
                    lambda x: f'{x:0{s}}')
            dm_model = None

        pbar.update(1)
        pbar.set_description("calculating modulation..".ljust(50))

        # calculate latent process using user-defined function "modulation"
        df_events_list = [
            _preprocess_event_latent_state(
                modulation, condition, df_events,
                _get_individual_params(
                    event_infos["subject"], all_individual_params)
            ) for df_events, event_infos in zip(df_events_list, event_infos_list)]

        df_events = pd.concat(df_events_list)
        pbar.update(1)
    else:
        pbar.update(2)
    ################################################################################
    # Get boldified signals.
    # this is done by utilizing nilearn.glm.first_level.hemodynamic_models.compute_regressor,
    # by providing 'onset','duration', and 'modulation' values.
    # the final matrix will be shaped as subject # x run # x n_scan
    # n_scane means the number of time points in fMRI data
    # if there is multiple session, still there would be no dimension indicating sessions info, 
    # but runs will be arranged as grouped by sessions number.
    # e.g. (subj-01, sess-01, run-01,:)
    #      (subj-01, sess-01, run-02,:)
    #                 ...
    #      (subj-01, sess-02, run-01,:)
    #      (subj-01, sess-02, run-02,:)
    #                 ...
    # this order should match with preprocessed fMRI image data.

    pbar.set_description("modulation signal making..".ljust(50))
    frame_times = t_r * (np.arange(n_scans) + t_r / 2)

    signals = []
    for name0, group0 in df_events.groupby(["subjID"]):
        signal_subject = []
        if n_session:
            for name1, group1 in group0.groupby(["session"]):
                for name2, group2 in group1.groupby(["run"]):
                    modulation_ = group2[["onset", "duration", "modulation"]].to_numpy().T
                    signal, _ = _make_boldify(modulation_, hrf_model, frame_times)
                    signal_subject.append(signal)
        else:
            for name1, group1 in group0.groupby(["run"]):
                modulation_ = group1[["onset", "duration", "modulation"]].to_numpy().T
                signal, _ = _make_boldify(modulation_, hrf_model, frame_times)
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
    pbar.update(1)

    ###########################################################################

    if save:
        np.save(sp / config.DEFAULT_MODULATION_FILENAME, signals)
    pbar.update(1)

    ###########################################################################
    # elapsed time check

    pbar.set_description("events preproecssing done!".ljust(50))

    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")

    return dm_model, df_events, signals, time_mask, layout


def _make_boldify(modulation_, hrf_model, frame_times):
    """
    Arguments:
        modulation_ (numpy.array): 
        hrf_model (str): 
        frame_times (numpy.array or list): 
        
    Return:
        boldified_signals (numpy.array): 
        
    """

    boldified_signals, name = compute_regressor(
        exp_condition=modulation_.astype(float),
        hrf_model=hrf_model,
        frame_times=frame_times)

    return boldified_signals, name


# todo: remove
def _get_individual_params(subject_id, all_individual_params):
    """
    Get individual parameter dictionary
    so the value can be referred by its name (type:str)

    Arguments:
        subject_id (int or str): subject ID number
        all_individual_params (pandas.DataFrame): pandas dataframe with individual parameter values where each row number matches with subject ID.

    Return:
        ind_pars (dict): individual parameter value. dictionary{parameter_name:value}

    """
    ind_pars = all_individual_params[
        all_individual_params["subjID"] == subject_id]

    return dict(ind_pars)


def _get_time_mask(condition, df_events, time_length, t_r, use_duration=False):
    """
    Get binary masked data indicating time points in use

    Arguments:
        condition: a function : row --> boolean, to indicate if use the row or not 
        df_events: dataframe for rows of one 'run' event data
        time_length: the length of target BOLD signal 
        t_r: time resolution
        use_duration: if True, use 'duration' column for masking, 
                      else use the gap between consecutive onsets as duration
    Return:
        time_mask: binary array.
                   shape: time_length
    """

    df_events = df_events.sort_values(by='onset')
    onsets = df_events['onset'].to_numpy()
    if use_duration:
        durations = df_events['duration'].to_numpy()
    else:
        durations = np.array(
            list(df_events['onset'][1:]) + [time_length * t_r]) - onsets

    mask = [condition(row) for _, row in df_events.iterrows()]
    time_mask = np.zeros(time_length)

    for do_use, onset, duration in zip(mask, onsets, durations):
        if do_use:
            time_mask[int(onset / t_r): int((onset + duration) / t_r)] = 1

    return time_mask


def _add_event_info(df_events, event_infos):
    """
    Add subject, run, session info to dataframe of events of single 'run' 

    Arguments:
        df_events: a dataframe for rows of one 'run' event data.
        event_infos: a dictionary containing  'subject', 'run', (and 'session' if applicable).

    Return:
        new_df: a dataframe with preprocessed rows
    """

    new_df = []
    df_events = df_events.sort_values(by='onset')

    def _add(row, info):
        row['subjID'] = info['subject']
        row['run'] = info['run']
        if 'session' in info.keys():
            row['session'] = info['session']  # if applicable

        return row

    for _, row in df_events.iterrows():
        new_df.append(_add(row, event_infos))

    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()

    return new_df


def _preprocess_event(preprocess, condition, df_events):
    """
    Preprocess dataframe of events of single 'run' 

    Arguments:
        preprocess: a funcion, which is converting row data to new one to match the name of value with hBayesDM.
                    preprocess must include the belows as the original event file would not have subject and run info.
                    row['subjID'] = info['subject'] 
                    row['run'] = f"{info['session']}_{info['run']}" (or row['run']=info['run'])
        condition: func : row --> boolean, to indicate if use the row or not.
        event_infos: a dictionary containing  'subject', 'run', (and 'session' if applicable).
        df_events: a dataframe for rows of one 'run' event data.

    Return:
        new_df: a dataframe with preprocessed rows
    """

    new_df = []
    df_events = df_events.sort_values(by='onset')

    for _, row in df_events.iterrows():
        if condition is not None and condition(row):
            new_df.append(preprocess(row))

    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()

    return new_df


def _preprocess_event_latent_state(modulation, condition, df_events, param_dict):
    """
    Aadd latent state value to for each row of dataframe of single 'run'

    Argumnets:
        modulation: a function, which is conducting row, param_dict --> row, function for calcualte latent state (or parameteric modulation value) 
        condition: a funcion, which is conducting row --> boolean, to indicate if use the row or not 
        df_events: dataframe for rows of one 'run' event data
        param_dict: a dictionary containing  model parameter value

    Return:
        new_df: a dataframe with latent state value
    """
    new_df = []
    df_events = df_events.sort_values(by='onset')

    for _, row in df_events.iterrows():
        if condition is not None and condition(row):
            new_df.append(modulation(row, param_dict))

    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()

    return new_df
    