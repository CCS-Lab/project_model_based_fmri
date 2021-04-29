#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

"""
helper functions for preprocessing behavior data
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


def _make_function_dfwise(function):
    
    def dfwise_function(df_events,**kwargs):
        new_df = []
        for _, row in df_events.iterrows():
            output = function(row,**kwargs)
            if isinstance(output,bool):
                # if output of function is boolean
                # it means that the given function is a filter.
                if output:
                    new_df.append(row)
            else:
                new_df.append(output)

        new_df = pd.concat(
            new_df, axis=1,
            keys=[s.name for s in new_df]
        ).transpose()
        return new_df
    
    return dfwise_function

def _make_single_time_mask(df_events, time_length, t_r,
                           use_duration=False):
    
    df_events = df_events.sort_values(by="onset")
    onsets = df_events["onset"].to_numpy()
    if use_duration:
        durations = df_events["duration"].to_numpy()
    else:
        durations = np.array(
            list(df_events["onset"][1:]) + [time_length * t_r]) - onsets
        
    time_mask = np.zeros(time_length)

    for onset, duration in zip(onsets, durations):
        time_mask[int(onset / t_r): int((onset + duration) / t_r)] = 1

    return time_mask

def _get_individual_param_dict(subject_id, individual_params):
    
    idp = individual_params[individual_params["subjID"] == subject_id]
    if len(idp) == 0:
        idp = individual_params[individual_params["subjID"] == int(subject_id)]
    if len(idp) == 0:
        return None
    idp = {k:d.item() for k,d in dict(idp).items()}
    return idp

def _boldify(modulation_, hrf_model, frame_times):
    
    boldified_signals, name = compute_regressor(
        exp_condition=modulation_.astype(float),
        hrf_model=hrf_model,
        frame_times=frame_times)

    return boldified_signals, name
