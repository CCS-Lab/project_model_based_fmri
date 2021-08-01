#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.04.29

from pathlib import Path
import numpy as np
import nibabel as nib
import json
from bids import BIDSLayout
import pandas as pd
from mbfmri.utils.descriptor import make_mbmvpa_description, version_diff
from mbfmri.utils import config # configuration for default names used in the package
from mbfmri.utils.plot import plot_data



class BIDSController():
    
    r"""
    
    Parameters
    ----------
    
    bids_layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout or BIDSController
        Root for input data. It should follow **BIDS** convention.
        
    subjects : list of str or "all", default="all"
        List of valid subject IDs. 
        If "all", all the subjects found in the layout will be loaded.
        
    sessions : list of str or "all", default="all"
        List of valid session IDs. 
        If "all", all the sessions found in the layout will be loaded.
            
    save_path : str or pathlib.PosixPath, default=None
        Path for saving preprocessed results. The MB-MVPA BIDS-like derivative layout will be created under the given path.
        If not input by the user, it will use "BIDSLayout_ROOT/derivatives/."
    
    fmriprep_name : str, default="fMRIPrep"
        Name of the derivative layout of fMRI preprocessed data.
    
    task_name : str, default=None
        Name of the task. If not given, the most common task name will be automatically selected.
    
    space_name : str, default=None
        Name of template space. If not given, the most common space in 
        input layout will be selected. 
        
    bold_suffix : str, default='regressors'
        Name of suffix indicating preprocessed fMRI file
        
    event_suffix : str, default='events'
        Suffix name for behavioral data file.

    confound_suffix : str, default='regressors'
        Name of suffix indicating confounds file
    
    ignore_original : boolean, default=True
        Indicate whether it would cover behaviroal data in the original BIDSLayout.
        If True, it will only consider data in the derivative layout for fMRI preprocessed data.
    
    ignore_fmriprep : boolean, default=False
        Indicate whether it can ignore fMRIPrep layout. 
        It should be True if users don't have fMRIPrep, but still 
        want to run computational modeling.
        
    t_r : float, default=None
        Time resolution in second. 
        It will be overrided by value from input data if applicable.
    
    slice_time_ref: float, default=.5
        Slice time reference in ratio in 0,1].
        It will be overrided by value from input data if applicable.
        
    Attributes
    ----------
    
    layout : bids.layout.layout.BIDSLayout
        (Original) BIDSLayout of input data. It should follow **BIDS** convention.
        The main data used from this layout is behaviroal data,``events.tsv``.
        
    fmriprep_layout : bids.layout.layout.BIDSLayout
        Derivative layout for fMRI preprocessed data. 
        ``fmriprep_layout`` is holding primarily preprocessed fMRI images (e.g. motion corrrected, registrated,...) 
        This package is built upon **fMRIPrep** by *Poldrack lab at Stanford University*. 
        
    mbmvpa_layout : ids.layout.layout.BIDSLayout
        Derivative layout for MB-MVPA. 
        The preprocessed voxel features and modeled latent process will be organized within this layout.
        
    voxelmask_path : pathlib.PosixPath
        Path for saving and locating integrated ROI mask. 
        
    meta_infos : pandas.DataFrame
        Dataframe of meta information of each run. 
        Following columns are included.
        ['subject', 'session', 'run', 'task', 'bold_path', 'confound_path', 'event_path', 't_r', 'n_scans','slice_time_ref']
    
    """
    
    
    def __init__(self,
                bids_layout,
                subjects='all',
                sessions='all',
                save_path=None,
                fmriprep_name="fMRIPrep",
                task_name=None,
                space_name=None,
                bold_suffix="bold",
                event_suffix="events",
                confound_suffix="regressors", 
                ignore_original=False,
                ignore_fmriprep=False,
                t_r=None,
                slice_time_ref=.5,):
        
        self.layout = self.get_base_layout(bids_layout)
        self.root = self.layout.root
        self.ignore_original = ignore_original
        self.ignore_fmriprep = ignore_fmriprep
        self.bold_suffix = bold_suffix
        self.event_suffix = event_suffix
        self.confound_suffix = confound_suffix
        self.fmriprep_name = fmriprep_name
        self.task_name = task_name
        self.nii_ext = config.NIIEXT
        self.space_name = space_name
        self.save_path = save_path
        self.mbmvpa_name = config.MBMVPA_PIPELINE_NAME
        self.subjects=subjects
        self.sessions=sessions
        self.t_r = t_r
        self.slice_time_ref =slice_time_ref
        self._set_fmriprep_layout()
        self._set_task_name()
        self._set_space_name()
        self._set_save_path()
        self._set_mbmvpa_layout()
        self._set_metainfo()
        self._set_voxelmask_path()
        
    def get_base_layout(self,bids_layout):
        
        # get BIDS layout which has events files
        print('INFO: start loading BIDSLayout')
        
        if isinstance(bids_layout,str) or isinstance(bids_layout,Path): 
            # input is given as path for bids layout root.
            layout = BIDSLayout(root=Path(bids_layout),derivatives=True)
        elif isinstance(bids_layout,BIDSLayout): # input is BIDS layout.
            layout = bids_layout
        elif isinstance(bids_layout, BIDSController): # input is already initiated.
            # assumed that BIDSController is already well initiated and just handed over.
            layout = BIDSLayout(root=bids_layout.root, derivatives=True)
        else:
            # not valid input
            assert False, ("please input BIDS root or BIDSLayout")
        return layout
    
    def _set_voxelmask_path(self,feature_name="unnamed"):
        self.voxelmask_path = Path(self.mbmvpa_layout.root)/ f"{config.DEFAULT_VOXEL_MASK_FILENAME}-{feature_name}.{self.nii_ext}"
    
    def _get_general_tr(self):
        if self.t_r is None:
            def _tr(layout):
                try:
                    return layout.get_tr()
                except:
                    return None
            t_rs = [_tr(l) for l in [self.layout,self.fmriprep_layout]]
            _t_rs= []
            for t_r in t_rs:
                if t_r is not None:
                    _t_rs.append(t_r)

            if len(_t_rs) == 0:
                print(f'INFO: No general TR is found. it will be searched for each run by json file.')
                return None
            else:
                return _t_rs[0]
        else:
            return self.t_r
        
            
    def _set_metainfo(self):
        
        print(f'INFO: target subjects-{self.subjects}')
        
        general_t_r = self._get_general_tr()
        general_slice_time_ref = self.slice_time_ref
        
        # set meta info for each run data in DataFrame format
        meta_infos = {'subject':[],        # subejct ID
                      'session':[],        # session ID
                      'run':[],            # run ID 
                      'task':[],           # task name
                      'bold_path':[],      # preprocessed bold file path 
                      'confound_path':[],  # regressor file path
                      'event_path':[],     # events file path
                      't_r':[],            # time resolution (sec)
                      'n_scans':[],        # number of scan (time dimension)
                      'slice_time_ref':[], # reference timing as rate of TR
                     }
        
        for file in self.get_bold_all():
            
            entities = file.get_entities()
            
            if self.subjects != 'all' and entities['subject'] not in self.subjects:
                continue
                
            if self.sessions != 'all' and \
                'session' in entities.keys() and\
                entities['session'] not in self.sessions:
                continue
                
            if 'session' in entities.keys(): 
                # if session is included in BIDS
                ses_id = entities['session']
            else:
                ses_id = None
                
            if 'run' in entities.keys(): 
                # if run is included in BIDS
                run_id = entities['run']
            else:
                run_id = None
                
            reg_file =self.get_confound(sub_id=entities['subject'],
                                             ses_id=ses_id,
                                             run_id=run_id,
                                             task_name=entities['task'])
            
            event_file =self.get_event(sub_id=entities['subject'],
                                             ses_id=ses_id,
                                             run_id=run_id,
                                             task_name=entities['task'])
            
            bold_file = self.get_bold(sub_id=entities['subject'],
                                             ses_id=ses_id,
                                             run_id=run_id,
                                             task_name=entities['task'])
            
            if not self.ignore_fmriprep and len(reg_file) < 1:
                # if regressor file is not found.
                continue
            if not self.ignore_fmriprep and len(bold_file) < 1:
                # if bold file is not found.
                continue
            if not self.ignore_original and len(event_file) < 1:
                # ignore_original means ignore events file.
                continue
            
            bold_path = bold_file[0].path if len(bold_file) >=1 else None
            reg_path = reg_file[0].path if len(reg_file) >=1 else None
            event_path = event_file[0].path if len(event_file) >=1 else None
            
            img_specs = self.fmriprep_layout.get(
                        return_type="file",
                        suffix=self.bold_suffix,
                        task=entities['task'],
                        extension=config.SPECEXT)
            if len(img_specs) > 0:
                img_spec = json.load(open( img_specs[0]))
                
                def _findany(d,key):
                    if key in d.keys():
                        return d[key]
                    else:
                        for _, inner_d in d.items():
                            if isinstance(inner_d,dict):
                                return _findany(inner_d,key)
                    return None
                
                spec_tr = _findany(img_spec,"RepetitionTime")
                spec_str = _findany(img_spec,"SliceTimingRef")
                if spec_tr is not None:
                    t_r = spec_tr
                else:
                    t_r = general_t_r
                if spec_str is not None:
                    slice_time_ref = spec_str
                else:
                    slice_time_ref = general_slice_time_ref
            else:
                t_r = general_t_r
                slice_time_ref = general_slice_time_ref
                
            # get n_scans
            n_scans = nib.load(bold_file[0].path).shape[-1]
            
            meta_infos['subject'].append(entities['subject'])
            meta_infos['session'].append(ses_id)
            meta_infos['run'].append(run_id)
            meta_infos['task'].append(entities['task'])
            meta_infos['bold_path'].append(bold_path)
            meta_infos['confound_path'].append(reg_path)
            meta_infos['event_path'].append(event_path)
            meta_infos['t_r'].append(t_r)
            meta_infos['n_scans'].append(n_scans)
            meta_infos['slice_time_ref'].append(slice_time_ref)
        
        self.meta_infos = pd.DataFrame(meta_infos)
        
        if not self.ignore_original:
            print(f'INFO: {len(self.meta_infos)} file(s) in Original & fMRIPrep.')
        else:
            print(f'INFO: {len(self.meta_infos)} file(s) in fMRIPrep.')
        
    def _set_mbmvpa_layout(self):
        if self.make_mbmvpa(self.save_path):
            print('INFO: MB-MVPA is newly set up.')
        
        if self.save_path is not None:
            self.mbmvpa_layout = BIDSLayout(root=self.save_path,validate=False)
        
        elif not self.mbmvpa_name in self.layout.derivatives.keys():
            self.layout.add_derivatives(path=self.save_path)
            self.mbmvpa_layout = self.layout.derivatives[self.mbmvpa_name]
            print('INFO: MB-MVPA is added as a new derivative')
            
        
        print('INFO: MB-MVPA is loaded')
        
    def _set_save_path(self):
        # set root path for MB-MVPA derivative layout
        if self.save_path is None:
            self.save_path = Path(self.root)/'derivatives'/config.DEFAULT_DERIV_ROOT_DIR
            
    def _set_task_name(self):
        # if task_name is not given, find the most common task name in the layout
        if self.task_name is None:
            print('INFO: task name is not designated. find most common task name')
            try:
                task_names = self.layout.get_task()
                task_name_lens = [len(self.layout.get(task=task_name,
                                                      suffix=self.bold_suffix)) for task_name in task_names]
                self.task_name = task_names[np.array(task_name_lens).argmax()]
            except:
                task_names = self.fmriprep_layout.get_task()
                task_name_lens = [len(self.fmriprep_layout.get(task=task_name,
                                                               suffix=self.bold_suffix)) for task_name in task_names]
                self.task_name = task_names[np.array(task_name_lens).argmax()]
            
            print('INFO: selected task_name is '+self.task_name)

    def _set_space_name(self):
        # if space_name is not given, find the most common space name in the layout
        if self.space_name is None:
            print('INFO: template space is not designated. find most common space')
            
            space_names = self.fmriprep_layout.get_space()
            space_name_lens = [len(self.fmriprep_layout.get(space=space_name,
                                                           suffix=self.bold_suffix)) for space_name in space_names]
            self.space_name = space_names[np.array(space_name_lens).argmax()]
            
            print('INFO: selected space_name is '+self.space_name)

        
    def _set_fmriprep_layout(self):
        if self.ignore_fmriprep:
            self.fmriprep_layout = self.layout
            return
        assert self.fmriprep_name in self.layout.derivatives.keys(), ("fmri prep. is not found")
        self.fmriprep_layout = self.layout.derivatives[self.fmriprep_name]
        self.fmriprep_version = self.fmriprep_layout.get_dataset_description()['PipelineDescription']['Version']
        if self.fmriprep_name == "fMRIPrep" and version_diff(self.fmriprep_version, "20.2.0") >= 0:
            self.confound_suffix = "timeseries"
        print('INFO: fMRIPrep is loaded')
        
    def summary(self):
        # print summary of loaded layout
        summaries = {}
        
        if self.fmriprep_layout is not None:
            fmriprep_summary = str(self.fmriprep_layout)
            fmriprep_pipeline_name = self.fmriprep_layout.description['PipelineDescription']['Name']
            summaries[fmriprep_pipeline_name] = fmriprep_summary
        else:
            summaries["fMRIPrep"] = "Not prepared or Ignored"
            
        mbmvpa_summary = str(self.mbmvpa_layout)
        mbmvpa_pipeline_name = self.mbmvpa_layout.description['PipelineDescription']['Name']
        summaries[mbmvpa_pipeline_name] = mbmvpa_summary
        
        summary_report = [f'[{pipeline_name:^12}] '+summary\
                            for pipeline_name, summary in summaries.items() ]
        
        summary_report = '\n'.join(summary_report)
        
        print(summary_report)
    
    def reload(self):
        '''reload MB-MVPA layout
        '''
        self.layout = BIDSLayout(root=self.root,derivatives=True)
            
        if not self.mbmvpa_name in self.layout.derivatives.keys():
            self.layout.add_derivatives(path=self.save_path)
        self.mbmvpa_layout = self.layout.derivatives[self.mbmvpa_name]
        
    def make_mbmvpa(self,mbmvpa_root):
        '''make MB-MVPA base layout if not exists
        '''
        mbmvpa_root = Path(mbmvpa_root)
        
        if not mbmvpa_root.exists():
            mbmvpa_root.mkdir()
        
        try:
            dataset_description = json.load(open(mbmvpa_root/'dataset_description.json'))
            bids_version = dataset_description["BIDSVersion"]
            assert dataset_description["PipelineDescription"]["Name"] == config.MBMVPA_PIPELINE_NAME
            return False
        except:
            if self.fmriprep_layout is not None:
                bids_version = self.fmriprep_layout.get_dataset_description()['BIDSVersion']
            else:
                bids_version = '1.1.1' # assumed

            make_mbmvpa_description(mbmvpa_root=mbmvpa_root,
                                bids_version=bids_version)
        return True
            
    def set_path(self, sub_id, ses_id=None):
        '''
        set & return path for saving MB-MVPA data
        path is defined as BIDS convention
        make directory if not exists
        '''
        sub_path = Path(self.mbmvpa_layout.root) / f'sub-{sub_id}'
        if not sub_path.exists():
            sub_path.mkdir()
        if ses_id is not None:
            ses_path = sub_path / f'ses-{ses_id}'
            if not ses_path.exists():
                ses_path.mkdir()
        else:
            ses_path = sub_path
        
        func_path = ses_path / 'func'
        if not func_path.exists():
            func_path.mkdir()
            
        return func_path
        
    def get_path(self, sub_id, ses_id=None):
        '''get path for saving directory of MB-MVPA of a single session
        '''
        if ses_id is not None:
            return Path(self.mbmvpa_layout.root)/f'sub-{sub_id}'/f'ses-{ses_id}'/'func'
        else:
            return Path(self.mbmvpa_layout.root)/f'sub-{sub_id}'/'func'
    
    
    def get_subjects(self):
        '''get subject ID list from BIDS
        '''
        return self.fmriprep_layout.get_subjects(task=self.task_name)
    
    def get_bold(self, sub_id=None, task_name=None, run_id=None, ses_id=None):
        '''get BOLD images from fMRIPrep BIDS
        '''
        return self.fmriprep_layout.get(
                        subject=sub_id, session=ses_id,
                        task=task_name,
                        run=run_id,
                        suffix=self.bold_suffix,
                        space=self.space_name,
                        extension=self.nii_ext)
        
    def get_event(self, sub_id=None, task_name=None, run_id=None, ses_id=None):
        '''get event files from BIDS
        '''
        return self.layout.get(
                        subject=sub_id, session=ses_id,
                        task=task_name,
                        run=run_id,
                        suffix=self.event_suffix,
                        extension=config.EVENTEXT)
    
    def get_event_all(self):
        '''get all of the event files from BIDS
        '''
        return self.layout.get(task=self.task_name,
                                suffix=self.event_suffix,
                                extension=config.EVENTEXT)
        
    def get_confound(self, sub_id=None, task_name=None, run_id=None, ses_id=None):
        '''get confound files from fMRIPrep BIDS
        '''
        return self.fmriprep_layout.get(
                    subject=sub_id, session=ses_id,
                    task=task_name,
                    run=run_id,suffix=self.confound_suffix,
                    extension=config.CONFOUNDEXT)
           
    def get_bold_all(self):
        '''get all of the BOLD images from fMRIPrep BIDS
        '''
        return self.fmriprep_layout.get(suffix=self.bold_suffix,
                                        space=self.space_name,
                                        task=self.task_name,
                                        extension=self.nii_ext)
    
    def get_confound_all(self):
        '''get all of the confound files from fMRIPrep BIDS
        '''
        return self.fmriprep_layout.get(suffix=self.confound_suffix,
                                        task=self.task_name,
                                        extension=config.CONFOUNDEXT)
        
    def save_voxelmask(self, voxel_mask):
        '''save voxel mask at voxel mask path.
        '''
        nib.save(voxel_mask, self.voxelmask_path)
        
        
    def plot_processed_data(self,
                            feature_name,
                            process_name,
                            h=10,
                            w=5,
                            fontsize=12):
        '''make and save plots of processed multi-voxel signals and latent process signals
        from MB-MVPA BIDS.
        '''
        process_name = ''.join(process_name.split('_'))
        save_path = Path(self.mbmvpa_layout.root)/f'plot_feature-{feature_name}_process-{process_name}'
        save_path.mkdir(exist_ok=True)
        
        n_plot = 0
        n_try = 0
        for _, row in self.meta_infos.iterrows():
            
            plotted = plot_data(mbmvpa_layout=self.mbmvpa_layout, 
                              subject=row['subject'],
                              run=row['run'],
                              feature_name=feature_name,
                              task_name= row['task'],
                              process_name=process_name,
                              session=row['session'],
                              t_r=row['t_r'],
                              w=w, 
                              h=h, 
                              fontsize=fontsize,
                              save=True,
                              save_path=save_path,
                              show=False)
            n_try += 1
            if plotted >0:
                n_plot += 1
                
        print(f'INFO: processed data [{n_plot}/{n_try}] are plotted for quality check.')
        
        
