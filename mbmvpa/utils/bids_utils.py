#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.04.29

"""
"""

from pathlib import Path
import numpy as np
import nibabel as nib
import json
from bids import BIDSLayout
import pandas as pd
from ..utils.descriptor import make_mbmvpa_description, version_diff
from ..utils import config # configuration for default names used in the package
from .plot import plot_data

import pdb


class BIDSController():
    
    r"""
    
    Parameters
    ----------
    
    bids_layout : str or pathlib.PosixPath or bids.layout.layout.BIDSLayout
        (Original) BIDSLayout of input data. It should follow **BIDS** convention.
        The main data used from this layout is behaviroal data,``events.tsv``.
    save_path : str or pathlib.PosixPath, default=None
        Path for saving preprocessed results. The MB-MVPA BIDS-like derivative layout will be created under the given path.
        If not input by the user, it will use "BIDSLayout_ROOT/derivatives/."
    fmriprep_name : str, default="fMRIPrep"
        Name of the derivative layout of fMRI preprocessed data.
    task_name : str, default=None
        Name of the task. If not given, the most common task name will be automatically selected.
    bold_suffix : str, default="bold"
        Suffix of filename indicating bold image data. Please refer to file naming convention in **BIDS** convention.
        ``sub-{}_task-{}_ses-{}_run-{}_space-{}_desc-preproc_bold.nii.gz`` is a typical bold image file.
    event_suffix : str, default,"events"
        Suffix of filename indicating behavioral data. Please refer to file naming convention in **BIDS** convention.
        ``sub-{}_task-{}_ses-{}_run-{}_events.tsv`` is a typical event file.
    confound_suffix : str, default="regressors"
        Suffix of filename indicating confounds or regressors data. Please refer to file naming convention in **BIDS** convention.
        Also refer to **fMRIPrep**.
        ``sub-{}_task-{}_ses-{}_run-{}_desc-confounds_regressors.tsv`` is a typical confounds file.
    ignore_original : boolean, default=False
        Indicator to tell whether it would cover behaviroal data in the original BIDSLayout ``layout``.
        If ``True``, it will only consider data in the derivative layout for fMRI preprocessed data,``fmriprep_layout``.
        
        
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
    root : str
        Root path of layout.
    save_path : pathlib.PosixPath
        Path for saving preprocessed results. The MB-MVPA BIDS-like derivative layout will be created under the given path.
        If not input by the user, it will use "BIDSLayout_ROOT/derivatives/."
    ignore_original : boolean
        Indicator to tell whether it would cover behaviroal data in the original BIDSLayout ``layout``.
        If ``True``, it will only consider data in the derivative layout for fMRI preprocessed data,``fmriprep_layout``.
    bold_suffix : str, default="bold"
        Suffix of filename indicating bold image data. Please refer to file naming convention in **BIDS** convention.
        ``sub-{}_task-{}_ses-{}_run-{}_space-{}_desc-preproc_bold.nii.gz`` is a typical bold image file.
    event_suuffix : str, default="events"
        Suffix of filename indicating behavioral data. Please refer to file naming convention in **BIDS** convention.
        ``sub-{}_task-{}_ses-{}_run-{}_events.tsv`` is a typical event file.
    confound_suffix : str, default="regressors"
        Suffix of filename indicating confounds or regressors data. Please refer to file naming convention in **BIDS** convention.
        Also refer to **fMRIPrep**.
        ``sub-{}_task-{}_ses-{}_run-{}_desc-confounds_regressors.tsv`` is a typical confounds file.
    fmriprep_name : str, default="fMRIPrep"
        Name of the derivative layout of fMRI preprocessed data. 
    mbmvpa_name : str, default="MB-MVPA"
        Name of the derivative layout of MB-MVPA data.
    task_name : str
        Name of the task. If not given, the most common task name will be automatically selected.
    voxelmask_path : pathlib.PosixPath
        Path for saving and locating integrated ROI mask. 
    meta_infos : pandas.DataFrame
        Dataframe of meta information of each run. 
        Following columns are included.
        ['subject', 'session', 'run', 'task', 'bold_path', 'confound_path', 'event_path', 't_r', 'n_scans']
    
        
    """
    
    
    def __init__(self,
                bids_layout,
                save_path=None,
                fmriprep_name="fMRIPrep",
                task_name=None,
                bold_suffix="bold",
                event_suffix="events",
                confound_suffix="regressors",
                ignore_original=False,
                ):
        
        self.layout = self.get_base_layout(bids_layout)
        self.root = self.layout.root
        self.ignore_original = ignore_original
        self.bold_suffix = bold_suffix
        self.event_suffix = event_suffix
        self.confound_suffix = confound_suffix
        self.fmriprep_name = fmriprep_name
        self.task_name = task_name
        self.save_path = save_path
        self.mbmvpa_name = config.MBMVPA_PIPELINE_NAME
        self._set_fmriprep_layout()
        self._set_task_name()
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
        self.voxelmask_path = Path(self.mbmvpa_layout.root)/ f"{config.DEFAULT_VOXEL_MASK_FILENAME}-{feature_name}.nii.gz"
            
    def _set_metainfo(self):
        
        # set meta info for each run data in DataFrame format
        meta_infos = {'subject':[],        # subejct ID
                      'session':[],        # session ID
                      'run':[],            # run ID 
                      'task':[],           # task name
                      'bold_path':[],      # preprocessed bold file path 
                      'confound_path':[],  # regressor file path
                      'event_path':[],     # events file path
                      't_r':[],            # time resolution (sec)
                      'n_scans':[]         # number of scan (time dimension)
                     }
        
        for bold_file in self.get_bold_all():
            
            entities = bold_file.get_entities()
            if 'session' in entities.keys(): 
                # if session is included in BIDS
                # old version of BIDS doesn't have it
                ses_id = entities['session']
            else:
                ses_id = None
            reg_file =self.get_confound(sub_id=entities['subject'],
                                             ses_id=ses_id,
                                             run_id=entities['run'],
                                             task_name=entities['task'])
            event_file =self.get_event(sub_id=entities['subject'],
                                             ses_id=ses_id,
                                             run_id=entities['run'],
                                             task_name=entities['task'])
            
            if len(reg_file) < 1:
                # if regressor file is not found.
                continue
            if not self.ignore_original and len(event_file) < 1:
                # ignore_original means ignore events file.
                continue
            
            # get t_r
            json_data = json.load(open(
                    self.fmriprep_layout.get(
                    return_type="file",
                    suffix=self.bold_suffix,
                    task=entities['task'],
                    extension="json")[0]))
            
            if "root" in json_data.keys():
                t_r = json_data["root"]["RepetitionTime"]
            else:
                t_r = json_data["RepetitionTime"]
            
            # get n_scans
            n_scans = nib.load(bold_file.path).shape[-1]
            
            meta_infos['subject'].append(entities['subject'])
            meta_infos['session'].append(ses_id)
            meta_infos['run'].append(entities['run'])
            meta_infos['task'].append(entities['task'])
            meta_infos['bold_path'].append(bold_file.path)
            meta_infos['confound_path'].append(reg_file[0].path)
            meta_infos['event_path'].append(event_file[0].path)
            meta_infos['t_r'].append(t_r)
            meta_infos['n_scans'].append(n_scans)
        
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
        
    def _set_fmriprep_layout(self):
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
        # reload MB-MVPA layout
        self.layout = BIDSLayout(root=self.root,derivatives=True)
            
        if not self.mbmvpa_name in self.layout.derivatives.keys():
            self.layout.add_derivatives(path=self.save_path)
        self.mbmvpa_layout = self.layout.derivatives[self.mbmvpa_name]
        
    def make_mbmvpa(self,mbmvpa_root):
        # make MB-MVPA base layout if not exists
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
        # set & return path for saving MB-MVPA data
        # path is defined as BIDS convention
        # make directory if not exists
        
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
        # get path for saving directory of MB-MVPA of a single session
        if ses_id is not None:
            return Path(self.mbmvpa_layout.root)/f'sub-{sub_id}'/f'ses-{ses_id}'/'func'
        else:
            return Path(self.mbmvpa_layout.root)/f'sub-{sub_id}'/'func'
    
    
    def get_subjects(self):
        return self.fmriprep_layout.get_subjects(task=self.task_name)
    
    def get_bold(self, sub_id=None, task_name=None, run_id=None, ses_id=None):
        return self.fmriprep_layout.get(
                        subject=sub_id, session=ses_id,
                        task=task_name,
                        run=run_id,
                        suffix=self.bold_suffix,
                        space=config.TEMPLATE_SPACE,
                        extension="nii.gz")
        
    def get_event(self, sub_id=None, task_name=None, run_id=None, ses_id=None):
        return self.layout.get(
                        subject=sub_id, session=ses_id,
                        task=task_name,
                        run=run_id,
                        suffix='events',
                        extension="tsv")
        
    def get_confound(self, sub_id=None, task_name=None, run_id=None, ses_id=None):
        return self.fmriprep_layout.get(
                    subject=sub_id, session=ses_id,
                    task=task_name,
                    run=run_id,suffix=self.confound_suffix,
                    extension="tsv")
           
    def get_bold_all(self):
        return self.fmriprep_layout.get(suffix=self.bold_suffix,
                                        space=config.TEMPLATE_SPACE,
                                        task=self.task_name,
                                        extension="nii.gz")
    
    def get_confound_all(self):
        return self.fmriprep_layout.get(suffix=self.confound_suffix,
                                        task=self.task_name,
                                        extension="tsv")
        
    def save_voxelmask(self, voxel_mask):
        nib.save(voxel_mask, self.voxelmask_path)
        
        
    def plot_processed_data(self,
                            feature_name,
                            process_name,
                            h=10,
                            w=5,
                            fontsize=12):
        
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
                              save_path=save_path)
            n_try += 1
            if plotted >0:
                n_plot += 1
                
        print(f'INFO: processed data [{n_plot}/{n_try}] are plotted for quality check.')
        
        
