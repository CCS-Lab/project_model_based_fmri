#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong, Cheoljun cho
@contact: mybirth0407@gmail.com
          cjfwndnsl@gmail.com
@last modification: 2020.11.16
"""

import os
from pathlib import Path
import bids
from bids import BIDSLayout, BIDSValidator

import pandas as pd
import numpy as np
from sklearn.preprocessing import minmax_scale
from scipy.stats import zscore

from nilearn.glm.first_level.hemodynamic_models import compute_regressor
import nibabel as nib
from tqdm import tqdm

import hbayesdm.models
import time

import logging

DEFAULT_SAVE_DIR = 'mvpa'

logging.basicConfig(level=logging.INFO)



"""

This part is implemented to preprocess "event" files spread out in BIDS layout to make a BOLD like data of latent process.

The latent process 



"""




################################################################################
"""
example functions for tom 2007 (ds000005)
"""

def example_prep_func_tom_mg(row, info):
    ## mandatory field ##
    row['subjID'] = info['subject']
    row['run'] = info['run'] 
    
    ## user defined mapping ##
    row['gamble'] = 1 if row['respcat'] == 1 else 0
    row['cert'] = 0
    
    return row

def example_cond_func_tom_mg(row):
    return True

def example_latent_func_piva_dd(row, info, param_dict):
    
    utility = (row['gain'] ** param_dict['rho']) \
            - (param_dict['lambda'] * (row['loss'] ** param_dict['rho']))
    row['modulation'] = utility
    
    return row
################################################################################

################################################################################
"""
example functions for piva 2019 (ds001882)
"""

def example_prep_func_piva_dd(row, info):
    ## mandatory field ##
    row['subjID'] = info['subject']
    row['run'] = info['run']
    row['session'] = info['session'] # if applicable
    
    ## user defined mapping ##
    if row['delay_left'] >= row['delay_right']:
        row['delay_later'] = row['delay_left']
        row['delay_sooner'] = row['delay_right']
        row['amount_later'] = row['money_left']
        row['amount_sooner'] = row['money_right']
        row['choice'] = 1 if row['choice'] == 1 else 0
    else:
        row['delay_later'] = row['delay_right']
        row['delay_sooner'] = row['delay_left']
        row['amount_later'] = row['money_right']
        row['amount_sooner'] = row['money_left']
        row['choice'] = 1 if row['choice'] == 2 else 0
        
    return row

def example_cond_func_piva_dd(row):
    return row['agent']==0

def example_latent_func_piva_dd(row, param_dict):
    ev_later = row['amount_later'] / (1 + param_dict['k'] * row['delay_later'])
    ev_sooner  = row['amount_sooner'] / (1 + param_dict['k'] * row['delay_sooner'])
    utility = ev_later - ev_sooner
    row['modulation'] = utility
    
    return row
################################################################################

def default_prep_func(row, info):
    ## mandatory field ##
    row['subjID'] = info['subject']
    row['run'] = info['run']
    if 'session' in info.keys():
        row['session'] = info['session'] # if applicable
    
    return row

def default_cond_func(row):
    return True

def _get_individual_params(subject_id, all_individual_params):
    # get individual parameter dictionary
    try:
        ind_pars = all_individual_params.loc[subject_id]
    except:
        ind_pars = all_individual_params.loc[int(subject_id)]
        
    return dict(ind_pars)


def _get_time_mask(condition, df_events, time_length, t_r, use_duration=False):
    # get binary mask indicating time points in use
    
    # condition : func : row --> boolean, to indicate if use the row or not 
    # df_events : dataframe for rows of one 'run' event data
    # time_length : the length of target BOLD signal 
    # t_r : time resolution
    # use_duration : boolean. if True, use 'duration' column for masking, 
                            # else use the gap between consecutive onsets as duration
        
    # return : time_mask : binary array. shape : time_length
    
    df_events = df_events.sort_values(by='onset')
    onsets = df_events['onset'].to_numpy()
    if use_duration:
        durations = df['duration'].to_numpy()
    else:
        durations = np.array(list(df_events['onset'][1:]) + [time_length * t_r]) - onsets
    
    mask = [condition(row) for _, row in df_events.iterrows()]
    time_mask = np.zeros(time_length)
    
    for do_use, onset, duration in zip(mask, onsets, durations):
        if do_use:
            time_mask[int(onset / t_r): int((onset + duration) / t_r)] = 1
        
    return time_mask


def _preprocess_event(preprocess, condition, df_events, event_infos, **kwargs):
    # preprocess dataframe of events of single 'run' 
    
    # preprocess : func : row --> row. converting row data to new one to match the name of value with hBayesDM.
                  # preprocess must include the belows as the original event file would not have subject and run info.
                            #row['subjID'] = info['subject'] 
                            #row['run'] = info['run']
                            #row['session'] = info['session']
    # condition : func : row --> boolean, to indicate if use the row or not 
    # event_infos : a dictionary containing  'subject', 'run', (and 'session' if applicable)
    # df_events : dataframe for rows of one 'run' event data
    
    # return : new_df, a dataframe with preprocessed rows
                                        
    
    new_df = []
    df_events = df_events.sort_values(by='onset')
    
    for _, row in df_events.iterrows():
        if condition is not None and condition(row):
            new_df.append(preprocess(row,event_infos, **kwargs))
    
    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()
    
    return new_df


def _preprocess_event_latent_state(latent_func, condition, df_events, param_dict):
    
    # add latent state value to for each row of dataframe of single 'run'  
    
    # latent_func : func : row, param_dict --> row, function for calcualte latent state (or parameteric modulation value) 
    # condition : func : row --> boolean, to indicate if use the row or not 
    # df_events : dataframe for rows of one 'run' event data
    # param_dict : a dictionary containing  model parameter value
    
    # return : new_df, a dataframe with latent state value
    
    new_df = []
    df_events = df_events.sort_values(by='onset')
    
    for _, row in df_events.iterrows():
        if condition is not None and condition(row):
            new_df.append(latent_func(row, param_dict))
    
    new_df = pd.concat(
        new_df, axis=1,
        keys=[s.name for s in new_df]
    ).transpose()
    
    return new_df

def preprocess_events(root, 
                      hrf_model="glover",
                      normalizer="minmax",
                      dm_model=None,
                      latent_func=None, 
                      layout=None,
                      preprocess=default_prep_func,
                      condition=lambda _: True,
                      df_events=None,
                      all_individual_params=None,
                      use_duration=False,
                      save=True,
                      save_path=None,
                      **kwargs # hBayesDM fitting 
                      ):
    """
    preprocessing event data to get BOLD-like signal and time mask for indicating valid range of data
    
    user can provide precalculated behaviral data through "df_events" argument, 
        which is the DataFrame with subjID, run, onset, duration, and modulation. (also session if applicable)
    
    if not, it will calculate latent process by using hierarchical Bayesian modeling using "hBayesDM" package. 
    
    user also can provide precalculated individual model parameter values, through "all_individual_params" argument. 
    
    ## parameters ##
    @root : root directory of BIDS layout
    @dm_model : model name specification for hBayesDM package. should be same as model name e.g. 'ra_prospect'
    @latent_func : user defined function for calculating latent process. f(single_row_data_frame, model_parameter) -> single_row_data_frame_with_latent_state
    @layout : BIDSLayout by bids package. if not provided, it will be obtained using root info.
    @preprocess : user defined function for modifying behavioral data. f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data
    @condition : user defined function for filtering behavioral data. f(single_row_data_frame) -> boolean
    @df_events : pd.DataFrame with 'onset', 'duration', 'modulation'. if not provided, it will be obtained by applyng hBayesDM modeling and user defined functions.
    @all_individual_params : pd.DataFrame with params_name columns and corresponding values for each subject if not provided, it will be obtained by fitting hBayesDM model
    @use_duration : if True use 'duration' column info to make time mask, if False regard gap between consecuting onsets as duration
    @hrf_model : specification for hemodynamic response function, which will be convoluted with event data to make BOLD-like signal
    @normalizer : normalization method to subject-wisely normalize BOLDified signal. "minimax" or "standard" 
    @save : boolean indicating whether save result if True, you will save y.npy, time_mask.npy and additionaly all_individual_params.tsv.
    @save_path : path for saving output. if not provided, BIDS root/derivatives/data will be set as default path
    
    ## return ##
    @dm_model : hBayesDM model.
    @df_events : integrated event DataFrame (preprocessed if not provided) with 'onset','duration','modulation'
    @signals : BOLD-like signal. shape : subject # x (session # x run #) x time length of scan x voxel #
    """

################################################################################
    pbar = tqdm(total=6)
    s = time.time()
    
    if save_path is None:
        sp = Path(layout.derivatives["fMRIPrep"].root) / DEFAULT_SAVE_DIR
    else:
        sp = Path(save_path)
    
    if save and not sp.exists():
        sp.mkdir()
        
################################################################################
# load bids layout

    if layout is None:
        pbar.set_description("loading bids dataset..".ljust(50))
        layout = BIDSLayout(root, derivatives=True)
    else:
        pbar.set_description("loading layout..".ljust(50))

    t_r = layout.get_tr()
    # this will aggregate all events file path in sorted way.
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
    
    session_exist = True
    if n_session:
        session_exist = False
        by = ["subjID", "run"]
        reshape_target = (n_subject, n_run, n_scans)
    else:
        by = ["subjID", "session", "run"]
        reshape_target = (n_subject, n_session, n_run, n_scans)

    # collecting dataframe data from event files in BIDS layout
    df_events_list = [event.get_df() for event in events]
    # event_info such as id number for subject, session, run 
    event_infos_list = [event.get_entities() for event in events]
    pbar.update(1)
################################################################################
    
    pbar.set_description("adjusting event file columns..".ljust(50))

    df_events_list = [
        _preprocess_event(
            preprocess, condition, df_events, event_infos
        ) for df_events, event_infos in zip(df_events_list, event_infos_list)
    ]
    pbar.update(1)
################################################################################
    
    pbar.set_description("calculating time mask..".ljust(50))
    
    time_masks = []

    for name, group in pd.concat(df_events_list).groupby(by):
        time_masks.append(_get_time_mask(condition, group , n_scans, t_r, use_duration))
    time_masks = np.array(time_masks).reshape(reshape_target)
    
    if save:
        np.save(sp / "time_masks.npy", time_masks)
        
    pbar.update(1)
    pbar.set_description("time mask preproecssing done!".ljust(50))
################################################################################

    if df_events is None: 
        
        # the case user does not provide precalculated bahavioral data
        # calculate latent process using user defined latent function
        
        assert(latent_func is not None)
        
        if all_individual_params is None: 
            
            # the case user does not provide individual model parameter values
            # obtain parameter values using hBayesDM package
            
            assert(dm_model is not None)
            
            pbar.set_description("hbayesdm doing (model: %s)..".ljust(50) % dm_model)
            dm_model = getattr(hbayesdm.models, dm_model)(
                data=pd.concat(df_events_list), **kwargs)
            pbar.update(1)
            all_individual_params = dm_model.all_ind_pars

            if save:
                all_individual_params.to_csv(sp / "all_individual_params.tsv", sep="\t")
                
        else:
            pbar.update(1)
        
        pbar.set_description("calculating modulation..".ljust(50))
        
        # calculate latent process using user defined latent function
        df_events_list =[
            _preprocess_event_latent_state(
                latent_func, condition, df_events,
                    _get_individual_params(
                        event_infos["subject"], all_individual_params)
                    ) for df_events, event_infos in zip(df_events_list, event_infos_list)]
        
        df_events = pd.concat(df_events_list)
        pbar.update(1)
    else:
        pbar.update(2)
################################################################################

    # get boldified signal
    # will be shaped as subject # x run # x n_scan
    # n_scane means the number of time points in fMRI data
    # if there is multiple session, still there would be no dimension indicating sessions info, 
    # but runs will be arranged as grouped by sessions number.
    # e.g. (subj-01, sess-01, run-01,:)
    #      (subj-01, sess-01, run-02,:)
    #                 ...
    #      (subj-01, sess-02, run-01,:)
    #      (subj-01, sess-02, run-02,:)
    #                 ...
    # this order should match with preprocessed fMRI image data
    
    pbar.set_description("modulation signal making..".ljust(50))
    frame_times = t_r * (np.arange(n_scans) + t_r / 2)

    signals = df_events[["onset", "duration", "modulation"]].to_numpy().T
    signals = signals.reshape(-1, n_scans)

    bold_signals = []
    for signal in signals:
        bold_signal, name = compute_regressor(
            exp_condition=signal,
            hrf_model=hrf_model,
            frame_times=frame_times)
        bold_signals.append(bold_signal)

    bold_signals = bold_signals.reshape(n_subject, -1)

    for i in range(bold_signals):
        if normalizer == "minmax":
            # using minimum and maximum values to make the value range in [-1,1]
            normalized_signal = minmax_scale(
                bold_signals[i], feature_range=(-1, 1), axis=0, copy=True
            )

        ## TODO ##
        if normalizer == "standard":
            normalized_signal = zscore(bold_signals[i], axis=None)
        else:
            normalized_signal = zscore(bold_signals[i], axis=None)
            bold_signals[i] = normalized_signal

    bold_signals = bold_signals.reshape(reshape_target)
    pbar.update(1)
################################################################################

    if save:
        np.save(sp / "y.npy", bold_signals)
        
    pbar.update(1)

################################################################################
# elapsed time check
    pbar.set_description("events preproecssing done!".ljust(50))
    
    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")

    return dm_model, df_events, bold_signals, time_masks
