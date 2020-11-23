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


logging.basicConfig(level=logging.INFO)


################################################################################
"""
example functions for tom 2007 (ds000005)
"""

def example_prep_func_tom_mg(row,info):

    row['subjID'] = info['subject']
    row['run'] = info['run']
    
    row['gamble'] = 1 if row['respcat'] ==1 else 0
    row['cert'] = 0
    
    return row


def example_cond_func_tom_mg(row):
    return True


params_name = ['row','lambda','tau']
param_rename = {'lambda':'_lambda'}
dm_model = 'ra_prospect'

def example_latent_func_piva_dd(row,info,rho, _lambda):
    
    utility = (row['gain'] ** rho) \
            - (_lambda * (row['loss'] ** rho))
    row['modulation'] = utility
    
    return row


################################################################################

################################################################################
"""
example functions for piva 2019 (ds001882)
"""
def example_prep_func_piva_dd(row,info):

    row['subjID'] = info['subject']
    row['run'] = f"{info['session']}_{info['run']}" # please incorporate session info if there is.
    
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

params_name = ['k','beta']
dm_model = 'dd_hyperbolic'

def example_latent_func_piva_dd(row,info,k,beta):
    
    ev_later   = row['amount_later'] / (1 + k * row['delay_later'])
    ev_sooner  = row['amount_sooner'] / (1 + k * row['delay_sooner'])
    utility = ev_later - ev_sooner
    row['modulation'] = utility
    
    return row
################################################################################


def _get_individual_params(subject_id, all_individual_params, param_names, param_rename):
    try:
        ind_pars = all_individual_params.loc[subject_id]
    except:
        ind_pars = all_individual_params.loc[int(subject_id)]
        
    return {param_rename[name] : ind_pars[name] for name in param_names}


def _get_time_mask(condition, df_events, time_length, t_r, use_duration=False):
    df_events = df_events.sort_values(by='onset')
    onsets = df_events['onset'].to_numpy()
    if use_duration:
        durations = df['duration'].to_numpy()
    else:
        durations = np.array(list(df_events['onset'][1:]) + [time_length * t_r]) - onsets
    
    mask = [condition(row) for row in df_events.rows()]
    time_mask = np.zeros(time_length)
    
    for do_use, onset, duration in zip(mask, onsets, durations):
        if do_use:
            time_mask[int(onset / t_r): int((onset + duration) / t_r)] = 1
        
    return time_mask


def _preprocess_event(preprocess, condition, df_events, event_infos, **kwargs):
    new_datarows = []
    df_events = df_events.sort_values(by='onset')
    
    for row in df_events.rows():
        if condition is not None and condition(row):
            new_datarows.append(preprocess(row,event_infos, **kwargs))
    
    new_datarows = pd.concat(
        new_datarows, axis=1,
        keys=[s.name for s in new_datarows]
    ).transpose()
    
    return new_datarows


def preprocess_events(root, 
                      hrf_model="glover",
                      normalizer="minmax",
                      dm_model=None,
                      latent_func=None, 
                      params_name=None,
                      param_rename=None,
                      layout=None,
                      preprocess=lambda x: x,
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
    
    ## parameters ##
    @root : root directory of BIDS layout
    @dm_model : model name specification for hBayesDM package. should be same as model name e.g. 'ra_prospect'
    @latent_func : user defined function for calculating latent process. f(single_row_data_frame, model_parameter) -> single_row_data_frame_with_latent_state
    @params_name : model parameter name specification. should be same as parameter in model, and latent_func arguments
    @param_rename : parameter re-naming if the pre defined name is incompatible with Python grammar. e.g. lambda is not allowed for argument name in python. 
    @layout : BIDSLayout by bids package. if not provided, it will be obtained using root info.
    @preprocess : user defined function for modifying behavioral data. f(single_row_data_frame) -> single_row_data_frame_with_modified_behavior_data
    @condition : user defined function for filtering behavioral data. f(single_row_data_frame) -> boolean
    @df_events : pd.DataFrame with 'onset', 'duration', 'modulation'. if not provided, it will be obtained by applyng hBayesDM modeling and user defined functions.
    @all_individual_params : pd.DataFrame with params_name columns and corresponding values for each subject if not provided, it will be obtained by fitting hBayesDM model
    @use_duration : if True use 'duration' column info to make time mask, if False regard gap between consecuting onsets as duration
    @hrf_model : specification for hemodynamic response function, which will be convoluted with event data to make BOLD-like signal
    @save : boolean indicating whether save result
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
        sp = Path(layout.derivatives["fMRIPrep"].root) / "data"
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
    events = layout.get(suffix="events", extension="tsv")
    image_sample = nib.load(
        layout.derivatives["fMRIPrep"].get(
            return_type="file",
            suffix="bold",
            extension="nii.gz")[0]
    )
    n_scans = image_sample.shape[-1]
    
    # collecting dataframe data from event files in BIDS layout
    df_events_list = [event.get_df() for event in events]
    # event_info such as id number for subject, session, run 
    event_infos_list = [event.get_entities() for event in events]
    pbar.update(1)
################################################################################
    
    # adjust each row of dataframe to using preprocess
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
    for name0, group0 in pd.concat(df_events_list).groupby(["subjID"]):
        time_mask_subject = []
        for name1, group1 in group0.groupby(["run"]):
            time_mask_subject.append(_get_time_mask(condition, group1 , n_scans, t_r, use_duration))
        time_masks.append(time_mask_subject)
        
    time_masks = np.array(time_masks)
    
    if save:
        np.save(sp / "time_mask.npy", time_masks)
        
    pbar.update(1)
    pbar.set_description("time mask preproecssing done!".ljust(50))
################################################################################

    if df_events is None:
        
        assert(latent_func is not None)
        assert(params_name is not None)
        
        if all_individual_params is None:
            
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

        if param_rename is None:
            param_rename = {}
        
        for name in params_name:
            if name not in param_rename.key():
                param_rename[name] = name
            
        df_events_list =[
            _preprocess_event(
                latent_func, condition, df_events, event_infos,
                    **_get_individual_params(
                        event_infos["subject"], all_individual_params,params_name, param_rename)
                    ) for df_events, event_infos in zip(df_events_list, event_infos_list)]
        
        df_events = pd.concat(df_events_list)
        pbar.update(1)
    else:
        pbar.update(2)
################################################################################

    pbar.set_description("modulation signal making..".ljust(50))
    frame_times = t_r * (np.arange(n_scans) + t_r/2)

    signals = []
    for name0, group0 in df_events.groupby(["subjID"]):
        signal_subject = []
        for name1, group1 in group0.groupby(["run"]):
            exp_condition = group1[["onset", "duration", "modulation"]].to_numpy().T
            exp_condition = exp_condition.astype(float)
            signal, name = compute_regressor(
                exp_condition=exp_condition,
                hrf_model=hrf_model,
                frame_times=frame_times)
            signal_subject.append(signal)
        
        signal_subject = np.array(signal_subject)
        reshape_target = signal_subject.shape
        
        if normalizer == "minmax":
            normalized_signal = minmax_scale(
                signal_subject.flatten(), feature_range=(-1, 1), axis=0, copy=True
            )

        if normalizer == "standard":
            normalized_signal = zscore(signal_subject.flatten(),axis=None)
        else:
            normalized_signal = zscore(signal_subject.flatten(),axis=None)
            
        normalized_signal = normalized_signal.reshape(-1, n_scans, 1)
        signals.append(normalized_signal)
    signals = np.array(signals)
    pbar.update(1)
################################################################################

    if save:
        np.save(sp / "y.npy", signals)
        
    pbar.update(1)

################################################################################
# elapsed time check
    pbar.set_description("events preproecssing done!".ljust(50))
    
    e = time.time()
    logging.info(f"time elapsed: {(e-s) / 60:.2f} minutes")
    
    
    return dm_model, df_events, signals, time_masks
