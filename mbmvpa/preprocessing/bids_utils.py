from pathlib import Path
from bids import BIDSLayout
frm ..utils.descriptor import make_mbmvpa_description, version_diff

from ..utils import config # configuration for default names used in the package


class BIDSController():
    
    def __init__(self,
                bids_layout,
                save_path=None,
                fmriprep_name="fMRIPrep",
                task_name=None,
                bold_suffix="bold",
                confound_suffix="regressors",
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
        else:
            # not valid input
            assert False, ("please input BIDS root or BIDSLayout")
        
        self.root = self.layout.root
        assert fmriprep_name in self.layout.derivatives.keys(), ("fmri prep. is not found")
        
        self.fmriprep_name = fmriprep_name
        self.fmriprep_layout = self.layout.derivatives[self.fmriprep_name]
        self.fmriprep_version = self.fmriprep_layout.get_dataset_description()['PipelineDescription']['Version']
        if self.fmriprep_name == "fMRIPrep" and version_diff(self.fmriprep_version, "20.2.0") >= 0:
            self.confound_suffix = "timeseries"
        else:
            self.confound_suffix = confound_suffix
        self.mbmvpa_name = config.PIPELINE_NAME
        self.bold_suffix = bold_suffix
        if task_name is None:
            self.task_name = self.fmriprep_layout.get_task()[0]
        else:
            self.task_name = task_name
        
        if save_path is None:
            save_path = Path(self.root)/'derivatives'/'mbmvpa'
            
        try:
            self.mbmvpa_layout = BIDSLayout(root=save_path, derivatives=True)
        except:
            self.make_mbmvpa(save_path)
            self.mbmvpa_layout = BIDSLayout(root=save_path, derivatives=True)
            
        self.make_mbmvpa(mbmvpa_root)
        self.n_subject, self.n_session, self.n_run, self.n_scans, self.t_r = self.get_metainfo()
        self.voxelmask_path = Path(self.mbmvpa_layout.root)/config.DEFAULT_VOXEL_MASK_FILENAME
        
    def make_mbmvpa(self,mbmvpa_root):
        
        if self.mbmvpa_name is not in self.layout.derivatives.keys():
            # mbmvpa directory in BIDS are not set
            mbmvpa_root = Path(self.root) /'derivatives'/ config.DEFAULT_DERIV_ROOT_DIR
            if not mbmvpa_root.exists():
                mbmvpa_root.mkdir()
            try:
                bids_version = self.fmriprep_layout.get_dataset_description()['BIDSVersion']
            else:
                bids_version = '1.1.1' # assumed
                
            make_mbmvpa_description(mbmvpa_root=mbmvpa_root,
                                bids_version=bids_version)
        else:
            return
            
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
    
    def get_metainfo(self):
        n_subject = len(self.fmriprep_layout.get_subjects())
        n_session = len(self.fmriprep_layout.layout.get_session())
        n_run = len(self.fmriprep_layout.get_run())

        image_sample = nib.load(
                    self.fmriprep_layout.get(
                    return_type="file",
                    suffix=self.bold_suffix,
                    task=self.task_name
                    extension="nii.gz")[0])
        
        n_scans = image_sample.shape[-1]
        try: 
            t_r = self.layout.get_tr()
        except:
            t_r = json.load(
                self.fmriprep_layout.get(
                return_type="file",
                suffix=self.bold_suffix,
                task=self.task_name
                extension="json")[0])["root"]["RepetitionTime"]
            t_r = float(t_r)
        
        return (n_subject, n_session, n_run, n_scans, t_r)
    
    def get_subjects(self):
        return self.fmriprep_layout.get_subjects(task=self.task_name)
    
    def get_boldfiles(self, sub_id, ses_id='.*', run_id='.*'):
        return self.fmriprep_layout.get(
                    subject=sub_id, session=ses_id, run=run_id,
                    return_type="file", suffix=self.bold_suffix,
                    extension="nii.gz", regex_search=True)
    
    def get_confoundfiles(self, sub_id, ses_id='.*', run_id='.*'):
        return self.layout.derivatives[self.fmriprep_name].get(
                    subject=sub_id, session=ses_id, run=run_id,
                    return_type="file", suffix=self.confound_suffix,
                    extension="tsv", regex_search=True)
    
    def save_voxelmask(self, voxel_mask):
        nib.save(voxel_mask, self.voxelmask_path)