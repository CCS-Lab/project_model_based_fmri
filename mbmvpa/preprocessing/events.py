#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2021.01.12
## class version 

"""
"""

from pathlib import Path

import hbayesdm.models
import nibabel as nib
import numpy as np
import pandas as pd
from .events_utils import _process_indiv_params, _add_event_info, _preprocess_event, \
                        _make_single_time_mask, _get_individual_param_dict, \
                        _add_latent_process_single_eventdata, _boldify
from .bids_utils import BIDSController
from bids import BIDSLayout
from tqdm import tqdm
from scipy.io import loadmat
import pdb

from ..utils import config # configuration for default names used in the package


class LatentProcessGenerator():
    def __init__(self, 
                  bids_layout,
                  bids_controller=None,
                  save_path=None,
                  task_name=None,
                  process_name="unnamed",
                  adjust_function=lambda x: x,
                  filter_function=lambda _: True,
                  latent_function=None,
                  modulation_dfwise=None,
                  dm_model="unnamed",
                  filter_for_modeling=None,
                  individual_params=None,
                  hrf_model="glover",
                  use_duration=False,
                  n_core=4,
                  ignore_original=False):

        # setting path informations and loading layout
        if bids_controller is None:
            self.bids_controller = BIDSController(bids_layout,
                                            save_path=save_path,
                                            task_name=task_name,
                                            ignore_original=ignore_original)
        else:
            self.bids_controller = bids_controller
        
        self.task_name = self.bids_controller.task_name
        # setting user-defined functions
        self.adjust_function = adjust_function
        self.filter_function = filter_function
        self.latent_function = latent_function
        self.modulation_dfwise = modulation_dfwise
        self.process_name = process_name
        assert "_" not in self.process_name
        
        # setting model fitting specification
        self.dm_model = dm_model
        self.filter_for_modeling = filter_for_modeling
        if individual_params is None:
            individual_params = Path(self.bids_controller.mbmvpa_layout.root)/ (
                f"task-{self.bids_controller.task_name}_model-{self.dm_model}_{config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME}")
        self.individual_params = _process_indiv_params(individual_params)

        # setting BOLD-like signal generating specification
        self.hrf_model = hrf_model
        self.use_duration = use_duration
        self.n_core=n_core
        
    def summary(self):
        self.bids_controller.summary()
        
    def _init_df_events_from_bids(self, adjust_function=None):

        if adjust_function is None:
            adjust_function = self.adjust_function
        
        df_events_list =[]
        event_infos_list = []
        
        for _, row in self.bids_controller.meta_infos.iterrows():
            df_events_list.append(pd.read_table(row['event_path']) )
            event_infos_list.append(dict(row))
        
        df_events_list = [
            _add_event_info(df_events, event_infos)
            for df_events, event_infos in zip(df_events_list, event_infos_list)
        ]
        
        if callable(adjust_function):
            # modify trial data by user-defined function "adjust_function"
            df_events_list = [
                _preprocess_event(
                    adjust_function, df_events
                ) for df_events, event_infos in zip(df_events_list, event_infos_list)
            ]
        return df_events_list, event_infos_list
    
    def _init_df_events_from_files(self, files, suffix="events",column_names=None, adjust_function=None):
        
        if adjust_function is None:
            adjust_function = self.adjust_function
        
        event_infos_list = []
        df_events_list = []
        for file in files:
            if suffix not in file:
                continue
            file = Path(file)
            event_info = {}
            for chunk in file.stem.split('_'):
                splits = chunk.split('-')
                if len(splits) == 2:
                    event_info[splits[0]] = splits[1]
            
            suffix = file.suffix
            if suffix == '.mat':
                df_events = loadmat(file)
                if column_names == None:
                    df_events = {key:data for key,data in df_events.items() if '__' not in key}
                else:
                    df_events = {key:df_events[key] for key in column_names}
                df_events = pd.Dataframe(df_events)
            elif suffix == '.tsv' or suffix =='.csv':
                df_events = df.read_table(file, sep='\t')
                if column_names is not None:
                    df_events = df_events[column_names]
            else:
                continue
                
            event_infos_list.append(info)
            df_events_list.append(df_events_list)
            
        df_events_list = [
            _add_event_info(df_events, event_infos)
            for df_events, event_infos in zip(df_events_list, event_infos_list)
        ]
        
        if callable(adjust_function):
            # modify trial data by user-defined function "adjust_function"
            df_events_list = [
                _preprocess_event(
                    adjust_function, df_events
                ) for df_events, event_infos in zip(df_events_list, event_infos_list)
            ]
        return df_events_list, event_infos_list
        
    def set_computational_model(self, 
                                overwrite=True,
                                dm_model=None, 
                                individual_params=None, 
                                df_events=None, 
                                adjust_function=None, 
                                filter_function=None,
                                **kwargs):

        if df_events is None:
            df_events,_ = self._init_df_events_from_bids(adjust_function=adjust_function)
            df_events= pd.concat(df_events)
            
        individual_params = _process_indiv_params(individual_params)
        if individual_params is None:
            individual_params = self.individual_params
            
        if dm_model is None:
            dm_model = self.dm_model
        if filter_function is None:
            if self.filter_for_modeling is None:
                filter_function = self.filter_function
            else :
                filter_function = self.filter_for_modeling

        if individual_params is None or overwrite:
            # the case user does not provide individual model parameter values
            # obtain parameter values using hBayesDM package

            assert dm_model != "unnamed", (
                "if df_events is None, must be assigned to dm_model.")

            df_events= pd.concat([df_events[[filter_function(row) \
                                for _, row in df_events.iterrows()]]])

            if type(dm_model) == str:
                print("INFO: running hBayesDM")
                if 'ncore' in kwargs.keys():
                    model = getattr(
                        hbayesdm.models, dm_model)(
                            data=df_events,
                            **kwargs)
                else:
                    model = getattr(
                        hbayesdm.models, dm_model)(
                            data=df_events,
                            ncore=self.n_core,
                            **kwargs)

            individual_params = pd.DataFrame(model.all_ind_pars)
            individual_params.index.name = "subjID"
            individual_params = individual_params.reset_index()
            individual_params_path = Path(self.bids_controller.mbmvpa_layout.root)/ (
                f"task-{self.bids_controller.task_name}_model-{self.dm_model}_{config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME}")
            individual_params.to_csv(individual_params_path,
                                     sep="\t", index=False)
            self._trained_dm_model = model

        self.individual_params = individual_params
        
        
    def run(self, overwrite=False, process_name=None):
        if process_name is None:
            process_name = self.process_name
            
        if self.individual_params is None:
            # computational model is not set yet.
            self.set_computational_model()
            
        task_name = self.task_name
        
        # this will aggregate all events file path in sorted way
        layout = self.bids_controller.layout
        
        df_events_list, event_infos_list = self._init_df_events_from_bids()
        
        print("INFO: indivudal parameters table")
        print(self.individual_params)
        print(f'INFO: start processing {len(df_events_list)} events.[task-{task_name}, process-{process_name}]')
        
        iterator = tqdm(zip(df_events_list, event_infos_list))
        for df_events, event_infos in iterator:
            sub_id = event_infos['subject']
            ses_id = event_infos['session'] if 'session' in event_infos.keys() else None
            run_id = event_infos['run']
            if ses_id is not None:
                prefix = f"sub-{sub_id}_task-{task_name}_ses-{ses_id}_run-{run_id}_desc-{process_name}"
                save_path = self.bids_controller.set_path(sub_id=sub_id, ses_id=ses_id)
            else:
                prefix = f"sub-{sub_id}_task-{task_name}_run-{run_id}_desc-{process_name}"
                save_path = self.bids_controller.set_path(sub_id=sub_id)
            
            timemask_path = save_path/(prefix+f'_{config.DEFAULT_TIMEMASK_SUFFIX}.npy')
            modulation_df_path = save_path/(prefix+f'_{config.DEFAULT_MODULATION_SUFFIX}.tsv')
            signal_path = save_path/(prefix+f'_{config.DEFAULT_SIGNAL_SUFFIX}.npy')
            
            if overwrite or not timemask_path.exists():
                timemask = _make_single_time_mask(self.filter_function, df_events, 
                                              event_infos['n_scans'], 
                                              event_infos['t_r'], 
                                              preprocess=self.adjust_function,
                                              use_duration=self.use_duration)
                np.save(timemask_path, timemask)
            
            if overwrite or not modulation_df_path.exists():
                param_dict = _get_individual_param_dict(sub_id, self.individual_params)
                if param_dict is None:
                    continue
                if self.modulation_dfwise is not None:
                    df_events = pd.concat([df_events[[self.filter_function(row) \
                                    for _, row in df_events.iterrows()]]])
                    df_events = df_events.sort_values(by="onset")
                    modulation_df = self.modulation_dfwise(df_events, param_dict)
                    modulation_df = modulation_df[['onset','duration','modulation']]
                else:
                    df_events = df_events.sort_values(by="onset")
                    modulation_df = _add_latent_process_single_eventdata(self.latent_function, self.filter_function,
                                                 df_events, param_dict, preprocess=self.adjust_function)
                modulation_df = modulation_df.astype({'modulation': 'float'})
                modulation_df.to_csv(modulation_df_path,
                                     sep="\t", index=False)
                
                
            if overwrite or not signal_path.exists():
                # assume slice time correction mid point.
                frame_times =  event_infos['t_r'] * \
                        np.arange(event_infos['n_scans']) + \
                          event_infos['t_r'] / 2.
            
                signal, _ = _boldify(
                    modulation_df.to_numpy(dtype=float).T, self.hrf_model, frame_times)
            
                np.save(signal_path, signal)
                
        print(f'INFO: events processing is done.')
        return 
    
    