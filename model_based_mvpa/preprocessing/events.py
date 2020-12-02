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
                      save=True,
                      scale=(-1, 1),
                      # hBayesDM fitting parameters
                      **kwargs,
                      ):
    """
    This function is for preprocessing behavior data ("events.tsv") to convert them to BOLD-like signals.
    Also, the time mask for indicating valid range of data will be obtained.
    User can provide precalculated behaviral data through "df_events" argument,
    which is the DataFrame with 'subjID', 'run', 'onset', 'duration', and 'modulation.' (also 'session' if applicable)
    If not, it will calculate latent process (or 'modulation') by using hierarchical Bayesian modeling by running "hBayesDM" package.
    User can also skip fitting a model by providing precalculated individual model parameter values, through "individual_params" argument.

    The BOLD-like signals will be used for a target(y) in MVPA.
    The time mask will be used for selecting time points in the data of both fMRI and target, which will be included in MVPA.

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
        save (boolean): if True, it will save "y.npy," "time_mask.npy" and additionaly "all_individual_params.tsv."
        scale (tuple(float, float)) : lower bound and upper bound for minmax scaling. will be ignored if 'standard' normalization is selected. default is -1 to 1.

    Return
        dm_model (hbayesdm.model): hBayesDM model.
        df_events (pandas.DataFrame): integrated event DataFrame (preprocessed if not provided) with 'onset','duration','modulation'
        signals (numpy.array): BOLD-like signals with shape: subject # x (session # x run #) x time length of scan x voxel #
        time_mask (numpy.array): a  binary mask indicating valid time point with shape: subject # x (session # x run #) x time length of scan
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
    
    # get meta info
    n_subject, n_session, n_run, n_scans, t_r = _get_metainfo(layout)
    
    
    events = layout.get(suffix="events", extension="tsv") # this will aggregate all events file path in sorted way
    df_events_list = [event.get_df() for event in events] # collecting dataframe from event files spread in BIDS layout
    event_infos_list = [event.get_entities() for event in events] # event_info contains ID number for subject, session, run
    
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

    df_events_list = _adjust_behavior_dataframes(preprocess,df_events_list,event_infos_list)
    
    pbar.update(1)

    ###########################################################################
    # get time masks

    pbar.set_description("calculating time masks..".ljust(50))

    time_mask = _get_total_time_mask(condition, df_events_list, time_length, t_r, use_duration)

    if save:
        np.save(sp / config.DEFAULT_TIME_MASK_FILENAME, time_mask) 

    pbar.update(1)

    ###########################################################################
    # Get dataframe with 'subjID','run','duration','onset','duration' and 'modulation' which are required fields for making BOLD-like signal
    # if user provided the "df_events" with those fields, this part will be skipped
    
    

    if df_events_custom is None: # the case user does not provide precalculated bahavioral data

        assert modulation is not None, "if df_events is None, must be assigned to latent_function"
        
        if condition_for_modeling is None:
            condition_for_modeling = condition
        
        # get individual parameter values in computational model which will be used to calculate the latent process('modulation').
        individual_params, dm_model = _get_individual_params(individual_params,dm_model,condition_for_modeling,df_events_list,**kwargs)
        
        pbar.update(1)
        pbar.set_description("calculating modulation..".ljust(50))
        
        # the 'modulation' values are obtained by applying user-defined function "modulation" with model parameter values
        df_events_ready = _add_latent_process_as_modulation(individual_params,modulation, condition, df_events_list, event_infos_list)
        
        pbar.update(1)
    else:
        # sanity check
        assert (
            ('modulation' in df_events_custom.columns)
            and ('subjID' in df_events_custom.columns)
            and ('run' in df_events_custom.columns)
            and ('onset' in df_events_custom.columns)
            and ('duration' in df_events_custom.columns)
            and ('modulation' in df_events_custom.columns)), ("missing column in behavior data")
        
        df_events_ready = df_events_custom
        pbar.update(2)
        
    ###########################################################################
    # Get boldified signals.

    pbar.set_description("modulation signal making..".ljust(50))
    signals = _convert_event_to_boldlike_signal(df_events_ready, t_r, hrf_model,normalizer)
    pbar.update(1)
    
    if save:
        np.save(sp / config.DEFAULT_MODULATION_FILENAME, signals)
    pbar.update(1)

    ###########################################################################
    # elapsed time check

    pbar.set_description("events preproecssing done!".ljust(50))

    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")

    return dm_model, df_events, signals, time_mask, layout





### helper functions ###



def _get_metainfo(layout):
    """
    Get meta information of fMRI experiment.
    
    Arguments:
        layout (nibabel.BIDSLayout): BIDSLayout by bids package.
        
    Returns:
        n_subject (int): the number of subjects
        n_session (int): the number of sessions
        n_run (int): the number of runs
        n_scans (int): the time length in a single run. 
        t_r (float): time resolution (second) of scanning
    """
    
    n_subject = len(layout.get_subjects())
    n_session = len(layout.get_session())
    n_run = len(layout.get_run())

    image_sample = nib.load(
        layout.derivatives["fMRIPrep"].get(
            return_type="file",
            suffix="bold",
            extension="nii.gz")[0]
    )
    n_scans = image_sample.shape[-1]
    t_r = layout.get_tr()
    
    return n_subject, n_session, n_run, n_scans, t_r

def _adjust_behavior_dataframes(preprocess,df_events_list,event_infos_list):
    """
    Adjust columns in events file
    
    Arguments:
        preprocess (func(pandas.Series, dict)-> pandas.Series)): a user-defined function for modifying each row of behavioral data. 
            - f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data check
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "evnts.tsv" files
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

def _get_indiv_param_dict(subject_id, individual_params):
    """
    Get individual parameter dictionary
    so the value can be referred by its name (type:str)

    Arguments:
        subject_id (int or str): subject ID number
        individual_params (pandas.DataFrame): pandas dataframe with individual parameter values where each row number matches with subject ID.

    Return:
        ind_pars (dict): individual parameter value. dictionary{parameter_name:value}

    """
    ind_pars = individual_params[
        individual_params["subjID"] == subject_id]

    return dict(ind_pars)

def _get_single_time_mask(condition, df_events, time_length, t_r, use_duration=False):
    """
    Get binary masked data indicating time points in use

    Arguments:
        condition (func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events (pandas.Daataframe): a dataframe retrieved from "evnts.tsv" file
        time_length (int): the length of target BOLD signal 
        t_r (float): time resolution
        use_duration (boolean): if True, use 'duration' column for masking, 
                      else use the gap between consecutive onsets as duration
    Return:
        time_mask (numpy.array): binary array with shape: time_length
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

def _get_total_time_mask(condition, df_events_list, time_length, t_r, use_duration=False):
    
    """
    Get binary masked data indicating time points in use
    
    binary mask indicating valid time points will be obtained by applying user-defined function "condition"
    "condition" function will censor each trial to decide whether include it or not
    if use_duration == True then 'duration' column data will be considered as a valid duration for selected trials,
    else the gap between consecutive trials will be used instead.

    Arguments:
        condition (func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "evnts.tsv" files
        time_length (int): the length of target BOLD signal 
        t_r (float): time resolution
        use_duration (boolean): if True, use 'duration' column for masking, 
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
                    time_mask_subject.append(_get_time_single_mask(
                        condition, group2, n_scans, t_r, use_duration))
        else:
            for name1, group1 in group0.groupby(["run"]):
                time_mask_subject.append(_get_time_single_mask(
                    condition, group1, n_scans, t_r, use_duration))

        time_mask.append(time_mask_subject)

    time_mask = np.array(time_mask)
    
    return time_mask


def _get_individual_params(individual_params,dm_model,condition_for_modeling,df_events_list,**kwargs):
    """
    Get individual parameter values of the model, either obtained from fitting hierarchical bayesian model supported by hBayesDM package or
    provided by user through the "individual_params" argument.
    
    Arguments:
        individual_params (str or Path or pandas.DataFrame) : pandas dataframe with params_name columns and corresponding values for each subject. if not provided, it will be obtained by fitting hBayesDM model
        dm_model (str or hbayesdm.model) : computational model by hBayesDM package. should be provided as the name of the model (e.g. 'ra_prospect') or a model object.
        condition_for_modeling (func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data which will be used for fitting computational model.
            - f(single_row_data_frame) -> True or False
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "evnts.tsv" files and preprocessed in the previous stage.
    
    Returns:
        individual_params (pandas.DataFrame): the dataframe containing parameter values for each subject
        dm_model (None or hbayesdm): the fitted model if fitting happens, otherwise, None
        
    """
    
    if individual_params is None:
        # the case user does not provide individual model parameter values
        # obtain parameter values using hBayesDM package

        assert dm_model is not None, "if df_events is None, must be assigned to dm_model."

        pbar.set_description(
            "hbayesdm doing (model: %s)..".ljust(50) % dm_model)
        
        df_events_list = [df_events[condition_for_modeling(df_events)] for df_events in df_events_list]
        
        if type(dm_model) == str:
            dm_model = getattr(
                hbayesdm.models, dm_model)(
                    data=pd.concat(df_events_list),
                    **kwargs)

        individual_params = dm_model.all_ind_pars
        cols = list(individual_params.columns)
        cols[0] = 'subjID'
        individual_params.columns = cols

        if save:
            individual_params.to_csv(
                sp / config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME,
                sep="\t")
    else:

        if type(individual_params) == str or type(individual_params) == type(Path()):
            individual_params = pd.read_csv(
                individual_params, sep="\t")
            s = len(str(individual_params["subjID"].max()))
            individual_params["subjID"] =\
                individual_params["subjID"].apply(
                    lambda x: f"{x:0{s}}")
        else:
            assert type(individual_params) == pd.DataFrame

        dm_model = None
        
    return individual_params, dm_model

def _add_latent_process_as_modulation(individual_params,modulation, condition, df_events_list, event_infos_list):
    """
    Calculate latent process using user-defined function "modulation", and add it to dataframe as a 'modulation' column.
    
    Arguments:
        individual_params (pandas.DataFrame): the dataframe containing parameter values for each subject
        modulation (func(pandas.Series, dict)-> Series): a user-defined function for calculating latent process (modulation). 
            - f(single_row_data_frame, model_parameter_dict) -> single_row_data_frame_with_latent_state 
        condition (func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events_list (list(pandas.DataFrame)): a list of dataframe retrieved from "evnts.tsv" files and preprocessed in the previous stage.
        event_infos_list (list(dict)): a list of dictionary containing information for each file. (ID number for subject, session, run)
    
    Returns:
        df_events_ready (pandas.DataFrame): a integrated dataframe with 'modulation' which is ready for BOLDification

    """
    
    df_events_list = [
            _preprocess_event_latent_state(
                modulation, condition, df_events,
                _get_indiv_param_dict(
                    event_infos["subject"], individual_params)
            ) for df_events, event_infos in zip(df_events_list, event_infos_list)]
    
    df_events_ready =  pd.concat(df_events_list)
    
    return df_events_ready
            
def _make_boldify(modulation_, hrf_model, frame_times):
    
    """
    BOLDify event data.
    
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

def _convert_event_to_boldlike_signal(df_events, t_r, hrf_model="glover",normalizer='minmax'):
    
    """
    BOLDify the preprocessed behavioral (event) data.
    
    this is done by utilizing nilearn.glm.first_level.hemodynamic_models.compute_regressor,
    by providing 'onset','duration', and 'modulation' values.
    the final matrix will be shaped as subject # x run # x n_scan
    n_scane means the number of time points in fMRI data
    if there is multiple session, still there would be no dimension indicating sessions info, 
    but runs will be arranged as grouped by sessions number.
    e.g. (subj-01, sess-01, run-01,:)
         (subj-01, sess-01, run-02,:)
                    ...
         (subj-01, sess-02, run-01,:)
         (subj-01, sess-02, run-02,:)
                    ...
    this order should match with preprocessed fMRI image data.
    
    Arguments:
        df_events (DataFrame): preprocessed dataframe which is ready to BOLDify 
        t_r (float): time resolution (second)
        hrf_model(str): name for hemodynamic model
        normalizer(str): method name for normalizing output BOLD-like signal. 'minmax' or 'standard' (=z_score)
        
    Return:
        boldified_signals (numpy.array): BOLD-like signal. 
        
    """
    
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
    
    return signals

def _add_event_info(df_events, event_infos):
    """
    Add subject, run, session info to dataframe of events of single 'run' 

    Arguments:
        df_events (pandas.Daataframe): a dataframe retrieved from "evnts.tsv" file
        event_infos (dict): a dictionary containing  'subject', 'run', (and 'session' if applicable).

    Return:
        new_df (pandas.Daataframe): a dataframe with event info
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


def _preprocess_event(preprocess, df_events):
    """
    Preprocess dataframe of events of single 'run' 

    Arguments:
        preprocess (func(pandas.Series, dict)-> pandas.Series)): a user-defined function for modifying each row of behavioral data. 
            - f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data check
        df_events (pandas.Daataframe): a dataframe retrieved from "evnts.tsv" file

    Return:
        new_df (pandas.Daataframe): a dataframe with preprocessed rows
    """

    new_df = []
    df_events = df_events.sort_values(by='onset')

    for _, row in df_events.iterrows():
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
        modulation (func(pandas.Series, dict)-> Series): a user-defined function for calculating latent process (modulation). 
            - f(single_row_data_frame, model_parameter_dict) -> single_row_data_frame_with_latent_state 
        condition (func(pandas.Series)-> boolean)): a user-defined function for filtering each row of behavioral data. 
            - f(single_row_data_frame) -> True or False
        df_events (pandas.DataFrame): a dataframe retrieved from "evnts.tsv" file and preprocessed in the previous stage.
        param_dict (dict): a dictionary containing model parameter value

    Return:
        new_df (pandas.DataFrame): a dataframe with latent state value ('modulation')
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
    