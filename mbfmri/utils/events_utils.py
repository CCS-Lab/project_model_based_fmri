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
import arviz as az
import matplotlib.pyplot as plt

from mbfmri.utils import config # configuration for default names used in the package

def _save_fitplots(model,
                    save_path,
                     credible_interval = 0.94,
                     point_estimate= 'mean',
                     bins = 'auto',
                     round_to = 2):
            
            # retrieved from https://github.com/CCS-Lab/hBayesDM/blob/develop/Python/hbayesdm/base.py

            if model.model_type == 'single':
                var_names = list(model.parameters_desc)
            else:
                var_names = ['mu_' + p for p in model.parameters_desc]

            axes = az.plot_posterior(model.fit,
                                     kind='hist',
                                     var_names=var_names,
                                     credible_interval=credible_interval,
                                     point_estimate=point_estimate,
                                     bins=bins,
                                     round_to=round_to,
                                     color='black')
            
            plt.savefig(Path(save_path)/f'plot_dist.png',bbox_inches='tight')
            for ax, (p, desc) in zip(axes, model.parameters_desc.items()):
                ax.set_title('{} ({})'.format(p, desc))
                
            az.plot_trace(model.fit, var_names=var_names)
            plt.savefig(Path(save_path)/f'plot_trace.png',bbox_inches='tight')

def _fit_dm_model(df_events,
                     dm_model,
                     n_core=4,
                     **kwargs):
        
        print(f"INFO: running computational model [hBayesDM-{dm_model}]")
        model = getattr(
                    hbayesdm.models, dm_model)(
                        data=df_events,
                        ncore=n_core,
                        **kwargs)
        
        
            
            
        var_names = ['mu_' + p for p in model.parameters_desc]
                
            
        individual_params = pd.DataFrame(model.all_ind_pars)
        individual_params.index.name = "subjID"
        individual_params = individual_params.reset_index()
        model_name = ''.join(dm_model.split('_'))
        
        return model, individual_params
    
    
def _get_looic(fit):
    inference = az.convert_to_inference_data(fit)
    inference['sample_stats']['log_likelihood'] = inference['posterior']['log_lik']
    return az.loo(inference).loo * (-2)

def _update_modelcomparison_table(table_path, model_name, value, criterion):
    to_add = pd.DataFrame({'model':[model_name],'value':[value], 'criterion':[criterion]})
    if Path(table_path).exists():
        table = pd.read_table(table_path)
        if model_name in list(table['model']):
            table = table[~((table['model']==model_name) & \
                            (table['criterion']==criterion))]
        table = pd.concat([table,to_add])
    else:
        table = pd.DataFrame({'model':[model_name],'value':[value], 'criterion':[criterion]})
    table.to_csv(table_path,sep="\t", index=False)
    

def _update_individual_params(individual_params_path,individual_params):
    if Path(individual_params_path).exists():
        old_params= pd.read_table(individual_params_path, converters={'subjID': str})
        if set(old_params.columns)==set(individual_params.columns):
            for subjID in list(individual_params['subjID']):
                old_params = old_params[~(old_params['subjID']==subjID)]
            individual_params = pd.concat([old_params,individual_params])
    individual_params.sort_values(by="subjID",inplace=True)
    return individual_params


def _process_indiv_params(individual_params):
    if type(individual_params) == str\
        or type(individual_params) == type(Path()):
        try:
            individual_params = pd.read_table(individual_params, converters={'subjID': str})
            print("INFO: individual parameters are loaded")
        except:
            individual_params = None
        
        return individual_params
    elif type(individual_params) == pd.DataFrame:
        return individual_params
    else:
        return None
    
def _add_event_info(df_events, event_infos,separate_run=False):

    new_df = []

    def _add(i, row, info):
        row["trial"] = i 
        if separate_run:
            if "session" in info.keys():
                row["subjID"] = info["subject"]+str(info["session"])+str(info["run"])
            else:
                row["subjID"] = info["subject"]+str(info["run"])
        else:
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
                           use_duration):
    
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

def _get_individual_param_dict(subject_id, ses_id,run_id, individual_params, separate_run):
    if separate_run:
        subject_id=str(subject_id)+str(ses_id)+str(run_id)
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
