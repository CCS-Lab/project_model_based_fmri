from pathlib import Path
import numpy as np
import nibabel as nib
import json
from bids import BIDSLayout
from ..utils.descriptor import make_mbmvpa_description, version_diff

from ..utils import config # configuration for default names used in the package

import pdb

class BIDSController():
    
    def __init__(self,
                bids_layout,
                save_path=None,
                fmriprep_name="fMRIPrep",
                task_name=None,
                bold_suffix="bold",
                confound_suffix="regressors",
                ignore_fmriprep=False
                ):
        
        # assumption 1 : only one task is considered. 
        # assumption 2 : all the settings of bold image are same.
        # assumption 3 : all the data are valid.
        
        if isinstance(bids_layout,str):
            self.layout = BIDSLayout(root=Path(bids_layout),derivatives=True)
        elif isinstance(bids_layout,Path):
            self.layout = BIDSLayout(root=bids_layout, derivatives=True)
        elif isinstance(bids_layout,BIDSLayout):
            self.layout = bids_layout
        elif isinstance(bids_layout, BIDSController):
            # assumed that BIDSController is already well initiated and just handed over.
            self.layout = BIDSLayout(root=bids_layout.root, derivatives=True)
        else:
            # not valid input
            assert False, ("please input BIDS root or BIDSLayout")
        
        self.root = self.layout.root
        
        self.ignore_fmriprep = ignore_fmriprep
        
        if not self.ignore_fmriprep:
            assert fmriprep_name in self.layout.derivatives.keys(), ("fmri prep. is not found")

            self.fmriprep_name = fmriprep_name
            self.fmriprep_layout = self.layout.derivatives[self.fmriprep_name]
            self.fmriprep_version = self.fmriprep_layout.get_dataset_description()['PipelineDescription']['Version']
            if self.fmriprep_name == "fMRIPrep" and version_diff(self.fmriprep_version, "20.2.0") >= 0:
                self.confound_suffix = "timeseries"
            else:
                self.confound_suffix = confound_suffix
            self.bold_suffix = bold_suffix
            
        else:
            self.fmriprep_name = fmriprep_name
            self.fmriprep_layout = None
            self.fmriprep_version = None
            self.confound_suffix = confound_suffix
            self.bold_suffix = bold_suffix
            
        self.mbmvpa_name = config.MBMVPA_PIPELINE_NAME
        
        if task_name is None:
            try:
                task_names = self.layout.get_task()
                task_name_lens = [len(self.layout.get(task=task_name,suffix=self.bold_suffix)) for task_name in task_names]
                self.task_name = task_names[np.array(task_name_lens).argmax()]
            except:
                task_names = self.fmriprep_layout.get_task()
                task_name_lens = [len(self.fmriprep_layout.get(task=task_name,suffix=self.bold_suffix)) for task_name in task_names]
                self.task_name = task_names[np.array(task_name_lens).argmax()]
            
        else:
            self.task_name = task_name
        
        if save_path is None:
            self.save_path = Path(self.root)/'derivatives'/config.DEFAULT_DERIV_ROOT_DIR
        else:
            self.save_path = save_path
            
        self.make_mbmvpa(self.save_path)
        
        if not self.mbmvpa_name in self.layout.derivatives.keys():
            self.layout.add_derivatives(path=self.save_path)
            
        self.mbmvpa_layout = self.layout.derivatives[self.mbmvpa_name]
            
        self.n_subject, self.n_session, self.n_run, self.n_scans, self.t_r = self.get_metainfo()
        self.voxelmask_path = Path(self.mbmvpa_layout.root)/config.DEFAULT_VOXEL_MASK_FILENAME
    
    def summary(self):
        
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
        self.layout = BIDSLayout(root=self.root,derivatives=True)
        if not self.ignore_fmriprep:
            self.fmriprep_layout = self.layout.derivatives[self.fmriprep_name]
            
        if not self.mbmvpa_name in self.layout.derivatives.keys():
            self.layout.add_derivatives(path=self.save_path)
        self.mbmvpa_layout = self.layout.derivatives[self.mbmvpa_name]
        
    def make_mbmvpa(self,mbmvpa_root):

        mbmvpa_root = Path(mbmvpa_root)
        
        if not mbmvpa_root.exists():
            mbmvpa_root.mkdir()
        
        try:
            dataset_description = json.load(mbmvpa_root/'dataset_description.json')
            bids_version = dataset_description["BIDSVersion"]
            assert dataset_description["PipelineDescription"]["Name"] == config.MBMVPA_PIPELINE_NAME
            
        except:
            if self.fmriprep_layout is not None:
                bids_version = self.fmriprep_layout.get_dataset_description()['BIDSVersion']
            else:
                bids_version = '1.1.1' # assumed

            make_mbmvpa_description(mbmvpa_root=mbmvpa_root,
                                bids_version=bids_version)
        
            
    def set_path(self, sub_id, ses_id=None):
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
        if ses_id is not None:
            return Path(self.mbmvpa_layout.root)/f'sub-{sub_id}'/f'ses-{ses_id}'/'func'
        else:
            return Path(self.mbmvpa_layout.root)/f'sub-{sub_id}'/'func'
    
    def get_metainfo(self,use_fmriprep_layout=True):
        if use_fmriprep_layout and self.fmriprep_layout is not None:
            layout = self.fmriprep_layout
        else:
            layout = self.layout
        n_subject = len(layout.get_subjects())
        n_session = len(layout.get_session())
        n_run = len(layout.get_run())
        image_sample = nib.load(
                    self.layout.get(
                    return_type="file",
                    suffix=self.bold_suffix,
                    task=self.task_name,
                    extension="nii.gz")[0])
        
        n_scans = image_sample.shape[-1]
        try: 
            t_r = self.layout.get_tr()
        except:
            t_r = json.load(
                layout.get(
                return_type="file",
                suffix=self.bold_suffix,
                task=self.task_name,
                extension="json")[0])["root"]["RepetitionTime"]
            t_r = float(t_r)
        
        return (n_subject, n_session, n_run, n_scans, t_r)
    
    def get_subjects(self):
        return self.fmriprep_layout.get_subjects(task=self.task_name)
    
    def get_bold(self, sub_id, run_id, ses_id=None):
        return self.fmriprep_layout.get(
                        subject=sub_id, session=ses_id, run=run_id, suffix=self.bold_suffix,
                        space=config.TEMPLATE_SPACE,
                        extension="nii.gz")
        
        
    def get_confound(self, sub_id, run_id, ses_id=None):
        return self.fmriprep_layout.get(
                    subject=sub_id, session=ses_id, run=run_id,suffix=self.confound_suffix,
                    extension="tsv")
           
    def get_bold_all(self):
        return self.fmriprep_layout.get(suffix=self.bold_suffix,space=config.TEMPLATE_SPACE,extension="nii.gz")
    
    def get_confound_all(self):
        return self.fmriprep_layout.get(suffix=self.confound_suffix,extension="tsv")
        
    def save_voxelmask(self, voxel_mask):
        nib.save(voxel_mask, self.voxelmask_path)
        
        
