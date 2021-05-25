#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho, Yedarm Seong
## contact: cjfwndnsl@gmail.com
## last modification: 2021.04.29

from pathlib import Path

import hbayesdm.models
import nibabel as nib
import numpy as np
import pandas as pd
from ..utils.events_utils import _process_indiv_params, _add_event_info, _make_function_dfwise, \
                                _make_single_time_mask, _get_individual_param_dict, _boldify
from ..utils.bids_utils import BIDSController
from bids import BIDSLayout
from tqdm import tqdm
from scipy.io import loadmat
import importlib

from ..utils import config # configuration for default names used in the package


class LatentProcessGenerator():
    r"""
    
    *LatetentPrecessGenerator* is for converting behavior data ("events.tsv") to BOLD-like signals,
    so the time dimension of them can match the time diemnsion of voxel feature data.
    The output files will be stored in the derivative BIDS layout for the package.
    You can expect a modulation tsv file with onset, duration, modulation, 
    a time mask npy file, a binary map for indicating valid time points, and a 
    BOLD-like signal npy file for each run of events.tsv files.
    
    Parameters
    ----------
    
    bids_layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout
        (Original) BIDSLayout of input data. It should follow **BIDS** convention.
        The main data used from this layout is behaviroal data,``events.tsv``.
    subjects : list of str or "all",default="all"
        List of subject IDs to load. 
        If "all", all the subjects found in the layout will be loaded.
    bids_controller : mbmvpa.utils.bids_utils.BIDSController, default=None
        BIDSController instance for controlling BIDS layout for preprocessing.
        If not given, then initiates the controller.
    save_path : str or pathlib.PosixPath, default=None
        Path for saving preprocessed results. The MB-MVPA BIDS-like derivative layout will be created under the given path.
        If not input by the user, it will use "BIDSLayout_ROOT/derivatives/."
    task_name : str, default=None
        Name of the task. If not given, the most common task name will be automatically selected.
    process_name : str, default="unnamed"
        Name of the target latent process.
        It should be match with the name defined in computational modeling
    adjust_function : function(pandas.Series, dict)-> pandas.Series, default=lambda x \: x
        User-defined row-wise function for modifying each row of behavioral data.
        *adjust_function* (a row of DataFrame) \: a row of DataFrame with modified behavior data
    filter_function : function(pandas.Series, dict)-> boolean, default=lambda \_ \: True
        User-defined row-wise function for filtering each row of behavioral data.
        *filter_function* (a row of DataFrame) \: True or False
    latent_function : function(pandas.Series, dict)-> pandas.Series, default=None
        User-defined row wise function for calculating latent process.
        The values will be indexed by 'modulation' column name.
        *latent_function* (a row of DataFrame)-> a row of DataFrame with modulation
    adjust_function_dfwise : function(pandas.DataFrame, dict)-> pandas.DataFrame, default=None
        User-defined dataframe-wise function for modifying each row of behavioral data.
        If not given, it will be made by using *adjust_function*.
        If given, it will override *adjust_function*.
    filter_function_dfwise : function(pandas.DataFrame, dict)-> pandas.DataFrame, default=None
        User-defined dataframe-wise function for filtering each row of behavioral data.
        If not given, it will be made by using *filter_function*.
        If given, it will override *filter_function*.
    latent_function_dfwise : function(pandas.DataFrame, dict)-> pandas.DataFrame, default=None
        User-defined dataframe-wise function for calculating latent process.
        If not given, it will be made by using *latent_function*.
        If given, it will override *latent_function*.
    computational_model : Object, default=None
        User-defined comutational model, which should include two callable methods.
        - fit : computational_model.fit(df_events) will conduct model fitting
        - get_parameters : computational_model.get_parameters() will get a Dataframe containing individual parameters.
    dm_model : str, default="unnamed"
        Name for computational modeling by **hBayesDM**. 
        You can still use this parameter to assign the name of the model, 
        even you would not choose to depend on hBayesDM.
    individual_params : str or pathlib.PosixPath or pandas.DataFrame, default=None
        Path or loaded DataFrame for tsv file with individual parameter values.
        If not given, find the file from the default path
        ``MB-MVPA_root/task-*task_name*_model-*model_name*_individual_params.tsv``
        If the path is empty, it will remain ``None`` indicating a need for running hBayesDM.
        So, it will be set after runniing hBayesDM package.
    hrf_model : str, default="glover"
        Name for hemodynamic response function, which will be convoluted with event data to make BOLD-like signal.
        The below notes are retrieved from the code of "nilearn.glm.first_level.hemodynamic_models.compute_regressor"
        (https://github.com/nilearn/nilearn/blob/master/nilearn/glm/first_level/hemodynamic_models.py)

        The different hemodynamic models can be understood as follows:
             - "spm": this is the hrf model used in SPM.
             - "spm + derivative": SPM model plus its time derivative (2 regressors).
             - "spm + time + dispersion": idem, plus dispersion derivative. (3 regressors)
             - "glover": this one corresponds to the Glover hrf.
             - "glover + derivative": the Glover hrf + time derivative (2 regressors).
             - "glover + derivative + dispersion": idem + dispersion derivative. (3 regressors)
    use_duration : boolean, default=False
        If True use "duration" column to make a time mask, 
        if False all the gaps following trials after valid trials would be included in the time mask.
    n_core : int, default=4
        Number of core in **hBayesDM**.
    ignore_original : boolean, default=False
        Indicator to tell whether it would cover behaviroal data in the original BIDSLayout ``layout``.
        If ``True``, it will only consider data in the derivative layout for fMRI preprocessed data,``fmriprep_layout``.
        And it means that the LatentProcessGenerator would not use events data in BIDSLayout, 
        rather it would use the list of files input by user.
        It will be used for initiating BIDSController.
    onset_name : str, default="onset"
        Column name indicating  *onset* values.
    duration_name : str, default="duration"
        Column name indicating *duration* values.
    end_name : str, default=None
        Column name indicating end of valid time.
        If given, *end*-*onset* will be used as *duration* and override *duration_name*.
        If ``None``, it would be ignored and *duration_name* will be used.
    use_1sec_duration : bool, default=True
        If True, *duration* will be fixed as 1 second.
        This parameter will override *duration_name* and *end_name*.
    kwargs : dict
        Dictionary for arguments for indicating additional parameters for running **hBayesDM**.
        
        
    """
    
    def __init__(self, 
                  bids_layout,
                  subjects="all",
                  bids_controller=None,
                  save_path=None,
                  task_name=None,
                  process_name="unnamed",
                  adjust_function=lambda x: x,
                  filter_function=lambda _: True,
                  latent_function=None,
                  adjust_function_dfwise=None,
                  filter_function_dfwise=None,
                  latent_function_dfwise=None,
                  computational_model=None,
                  dm_model="unnamed",
                  individual_params=None,
                  hrf_model="glover",
                  use_duration=False,
                  n_core=4,
                  ignore_original=False,
                  ignore_fmriprep=False,
                  onset_name="onset",
                  duration_name="duration",
                  end_name=None,
                  use_1sec_duration=True,
                  skip_compmodel=False,
                  **kwargs):

        # set path informations and load layout
        if bids_controller is None:
            self.bids_controller = BIDSController(bids_layout,
                                            subjects=subjects,
                                            save_path=save_path,
                                            task_name=task_name,
                                            ignore_original=ignore_original)
        else:
            self.bids_controller = bids_controller
        
        self.task_name = self.bids_controller.task_name
        
        # setting dataframe-wise functions
        if adjust_function_dfwise is not None:
            self.adjust_function_dfwise = adjust_function_dfwise
        else:
            self.adjust_function_dfwise = _make_function_dfwise(adjust_function)
            
        if filter_function_dfwise is not None:
            self.filter_function_dfwise = filter_function_dfwise
        else:
            self.filter_function_dfwise = _make_function_dfwise(filter_function)
            
        if latent_function_dfwise is not None or skip_compmodel:
            self.latent_function_dfwise = latent_function_dfwise
        else:
            self.latent_function_dfwise = _make_function_dfwise(latent_function)
            
        self.process_name = process_name
        assert "_" not in self.process_name, (" '_'should not be included in the process name.")
        
        # setting model fitting specification
        self.dm_model = dm_model
        
        self.skip_computational_modeling = skip_compmodel
        
        if self.skip_computational_modeling:
            self.individual_params = config.IGNORE_INDIV_PARAM
        else:
            if individual_params is None:
                model_name = ''.join(dm_model.split('_'))
                individual_params = Path(self.bids_controller.mbmvpa_layout.root)/ (
                    f"task-{self.bids_controller.task_name}_model-{model_name}_{config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME}")
            self.individual_params = _process_indiv_params(individual_params)

        # setting BOLD-like signal generating specification
        self.hrf_model = hrf_model
        self.use_duration = use_duration
        self.n_core=n_core
        self.onset_name=onset_name
        self.duration_name=duration_name
        self.end_name=end_name
        self.use_1sec_duration = use_1sec_duration
        self.computational_model = computational_model
        
        
    def summary(self):
        self.bids_controller.summary()
        
    def _init_df_events_from_bids(self):

        df_events_list =[]
        event_infos_list = []
        
        # aggregate events dataframe using bids_controller
        for _, row in self.bids_controller.meta_infos.iterrows():
            df_events_list.append(pd.read_table(row['event_path']) )
            event_infos_list.append(dict(row))
        
        # add meta info to events data
        df_events_list = [
            _add_event_info(df_events, event_infos)
            for df_events, event_infos in zip(df_events_list, event_infos_list)
        ]
        
        return df_events_list, event_infos_list
    
    def _init_df_events_from_files(self, files, suffix="tsv",column_names=None):
        
        # aggregate events dataframe from input files
        event_infos_list = []
        df_events_list = []
        for file in files:
            if suffix not in file:
                continue
            file = Path(file)
            
            # parse file name and make meta-info
            # assume BIDS format like 'key1-name1_key2-name2_...'
            event_info = {}
            for chunk in file.stem.split('_'):
                splits = chunk.split('-')
                if len(splits) == 2:
                    event_info[splits[0]] = splits[1]
            
            # load events data according to file type
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
            
        # add meta info to events datav
        df_events_list = [
            _add_event_info(df_events, event_infos)
            for df_events, event_infos in zip(df_events_list, event_infos_list)
        ]
        return df_events_list, event_infos_list
        
    def set_computational_model(self, 
                                overwrite=True,
                                individual_params=None, 
                                df_events=None, 
                                adjust_function_dfwise=None, 
                                filter_function_dfwise=None,
                                n_core=None,
                                computational_model=None,
                                **kwargs):
            
        if n_core is None:
            n_core = self.n_core
            
        individual_params = _process_indiv_params(individual_params)
        
        if individual_params is None:
            individual_params = self.individual_params
        
        if computational_model is None:
            computational_model = self.computational_model
            
        dm_model = self.dm_model
        
        if adjust_function_dfwise is None:
            adjust_function_dfwise = self.adjust_function_dfwise
        
        if filter_function_dfwise is None:
            filter_function_dfwise = self.filter_function_dfwise
            

        if individual_params is None or overwrite:
            # the case user does not provide individual model parameter values
            # obtain parameter values using hBayesDM package

            assert dm_model != "unnamed", (
                "if df_events is None, must be assigned to dm_model.")
            
            if df_events is None:
                df_events_list,_ = self._init_df_events_from_bids()
                df_events_list = [filter_function_dfwise(
                                        adjust_function_dfwise(df_events)) for df_events in df_events_list]
                df_events_list = [df_events.sort_values(by=self.onset_name) for df_events in df_events_list]
                df_events= pd.concat(df_events_list)
            
            if computational_model is not None:
                print("INFO: running computational model [user-defined]")
                model.fit(df_events)
                individual_params = model.get_parameters()
                
            elif type(dm_model) == str:
                print(f"INFO: running computational model [hBayesDM-{dm_model}]")
                model = getattr(
                    hbayesdm.models, dm_model)(
                        data=df_events,
                        ncore=self.n_core,
                        **kwargs)
                
                individual_params = pd.DataFrame(model.all_ind_pars)
                individual_params.index.name = "subjID"
                individual_params = individual_params.reset_index()
                model_name = ''.join(dm_model.split('_'))
                
                
            individual_params_path = Path(self.bids_controller.mbmvpa_layout.root)/ (
                f"task-{self.bids_controller.task_name}_model-{model_name}_{config.DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME}")

            # save indiv params
            individual_params.to_csv(individual_params_path,
                                     sep="\t", index=False)
            self._trained_dm_model = model

        self.individual_params = individual_params
        
        
    def run(self,
            overwrite=True,
            process_name=None,
            modeling_kwargs={},
            **kwargs):
        
        if process_name is None:
            process_name = self.process_name
            
        if self.individual_params is None:
            # computational model is not set yet.
            self.set_computational_model(**modeling_kwargs)
            
        task_name = self.task_name
        
        # this will aggregate all events file path in sorted way
        layout = self.bids_controller.layout
        
        # get data from BIDS
        df_events_list, event_infos_list = self._init_df_events_from_bids()
        
        print("INFO: indivudal parameters table")
        print(self.individual_params)
        print(f'INFO: start processing {len(df_events_list)} events.[task-{task_name}, process-{process_name}]')
        
        iterator = tqdm(zip(df_events_list, event_infos_list))
        
        for df_events, event_infos in iterator:
            
            # manipulate dataframe
            df_events = self.filter_function_dfwise(
                                self.adjust_function_dfwise(df_events))
            df_events = df_events.sort_values(by=self.onset_name)
            
            
            # set save path for timemaske, modulation_df and signal
            sub_id = event_infos['subject']
            ses_id = event_infos['session'] if 'session' in event_infos.keys() else None
            run_id = event_infos['run']
            prefix = f"sub-{sub_id}_task-{task_name}"
            if ses_id is not None:
                prefix += f"_ses-{ses_id}"
            if run_id is not None:
                prefix += f"_run-{run_id}"
            prefix += f"_desc-{process_name}"
            save_path = self.bids_controller.set_path(sub_id=sub_id, ses_id=ses_id)
            
            timemask_path = save_path/(prefix+f'_{config.DEFAULT_TIMEMASK_SUFFIX}.npy')
            modulation_df_path = save_path/(prefix+f'_{config.DEFAULT_MODULATION_SUFFIX}.tsv')
            signal_path = save_path/(prefix+f'_{config.DEFAULT_SIGNAL_SUFFIX}.npy')
            
            # get dataframe with onset, duration and modulation
            # these 3 columns are required to make HRF convoluted signals
            if self.onset_name != "onset":
                df_events["onset"] = df_events[self.onset_name]
            if self.use_1sec_duration:
                df_events["duration"] = 1
            elif self.end_name is not None:
                df_events["duration"] = df_events[self.end_name]-df_events["onset"]
            elif self.duration_name != "duration":
                df_events["duration"] = df_events[self.duration_name]
            
            # get & save time mask
            if overwrite or not timemask_path.exists():
                timemask = _make_single_time_mask(df_events, 
                                              event_infos['n_scans'], 
                                              event_infos['t_r'],
                                              use_duration=self.use_duration)
                np.save(timemask_path, timemask)
            
            # get & save latent process (modulation) dataframe
            
            if overwrite or not modulation_df_path.exists():
                
                if self.skip_computational_modeling:
                    df_events['modulation'] = df_events[self.process_name]
                    modulation_df = df_events
                else:
                    param_dict = _get_individual_param_dict(sub_id, self.individual_params)
                    if param_dict is None:
                        continue
                    modulation_df = self.latent_function_dfwise(df_events,param_dict=param_dict)
                    
                modulation_df = modulation_df[['onset','duration','modulation']]
                modulation_df = modulation_df.astype({'modulation': 'float'})
                modulation_df = modulation_df.sort_values(by="onset")
                modulation_df.to_csv(modulation_df_path,
                                     sep="\t", index=False)
                
            # get & save BOLD-like signals
            if overwrite or not signal_path.exists():
                # assume slice time correction at mid. points of scans.
                frame_times =  event_infos['t_r'] * \
                        np.arange(event_infos['n_scans']) + \
                          event_infos['t_r'] / 2.
                signal, _ = _boldify(
                    modulation_df.to_numpy(dtype=float).T, self.hrf_model, frame_times)
            
                np.save(signal_path, signal)
                
        print(f'INFO: events processing is done.')
        return 