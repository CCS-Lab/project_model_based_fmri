#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.05.03

from ..preprocessing.events import LatentProcessGenerator
from ..preprocessing.bold import VoxelFeatureGenerator
from ..data.loader import BIDSDataLoader
from ..models.mvpa_general import MVPA_CV
from ..utils.config import DEFAULT_ANALYSIS_CONFIGS
from ..utils.report import build_elasticnet_report_functions, build_base_report_functions
import yaml, importlib, copy

# dictionary for module path and class name for implemented model.
MVPA_MODEL_DICT = {'elasticnet':['mbmvpa.models.elasticnet','MVPA_ElasticNet'],
                   'mlp':['mbmvpa.models.tf_mlp','MVPA_MLP'],
                   'cnn':['mbmvpa.models.tf_cnn','MVPA_CNN']}

# dictionary for report function for implemented model.
MVPA_REPORT_DICT = {'elasticnet':build_elasticnet_report_functions,
                   'mlp':build_base_report_functions,
                   'cnn':build_base_report_functions}

NEED_RECONSTRUCT_MODEL = ['cnn']

def run_mbmvpa(config=None,
              mvpa_model='elasticnet',
              report_path='.',
              overwrite=False,
              **kwargs):
    
    # callable wrapper of MBMVPA
    
    mbmvpa = MBMVPA(config=config,
                     mvpa_model=mvpa_model,
                     report_path=report_path,
                     **kwargs)

    return mbmvpa.run(overwrite=overwrite)
    

class MBMVPA():
    
    r"""
    
    A wrapper of functions in MB-MVPA package to enable a single line usage.
    The following procedures are done by MBMVPA class.

    1. preprocess fMRI & behavioral data
    2. load preprocessed data
    3. fit MVPA models and interprete the models

    By running this code, users can expect to get a brain activation pattern attributed to
    the target latent process defined in the computational model. 
    
    Parameters
    ----------
    
    config : dict or str or pathlib.PosixPath, default=None
        dictionary for keyworded configuration, or path for yaml file.
        The configuration input will override the default configuration.
    mvpa_model : str, default="elasticnet"
        name for MVPA model. Currently, "elasticnet," "mlp" and "cnn" are allowed.
    report_path : str or pathlib.PosixPath, defualt="."
        path for saving outputs of MVPA_CV module. 
        please refer to mbmvpa.models.mvpa_general.MVPA_CV
    **kwargs : dict
        dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
    """
    def __init__(self,
                 config=None,
                 mvpa_model='elasticnet',
                 report_path='.',
                 **kwargs):
        
        # load & set configuration
        self.config = DEFAULT_ANALYSIS_CONFIGS
        self._override_config(config)
        self._add_kwargs_to_config(kwargs)
        self.config['MVPACV']['cv_save_path']=report_path
        
        # find the function for calculating the target latent process
        # if the function is not given, then it will find from implemented models
        # please refer to the document for available models and latent processes
        if 'latent_function' not in self.config['LATENTPROCESS'].keys():
            self._add_latent_info_kwargs(self.config['LATENTPROCESS']['dm_model'],
                                         self.config['LATENTPROCESS']['process_name'], 
                                         self.config['LATENTPROCESS'])
        
        # setting name for saving outputs
        self.mvpa_model_name = mvpa_model
        result_name = '-'.join([''.join(self.config['LATENTPROCESS']['dm_model'].split('_')),
                                self.config['LOADER']['task_name'],
                                self.config['LOADER']['process_name'],
                                self.config['LOADER']['feature_name']])
        self.config['MVPAREPORT'][self.mvpa_model_name]['experiment_name'] = result_name
        self.config['MVPACV']['experiment_name'] = result_name
        
        # initiating internal modules for preprocessing input data
        self.X_generator = VoxelFeatureGenerator(**self.config['VOXELFEATURE'])
        self.bids_controller = self.X_generator.bids_controller
        self.y_generator = LatentProcessGenerator(bids_controller=self.bids_controller,
                                                  **self.config['LATENTPROCESS'])
        self.loader = None
        
        # find MVPA model and report function
        self._mvpa_model_class = getattr(
                                    importlib.import_module(MVPA_MODEL_DICT[mvpa_model][0]),
                                    MVPA_MODEL_DICT[mvpa_model][1])
        self._mvpa_report_func = MVPA_REPORT_DICT[mvpa_model]
        
        # set flag if reconstructing voxel feature (1D) to 4D is required
        if self.mvpa_model_name in NEED_RECONSTRUCT_MODEL:
            self.config['LOADER']['reconstruct'] = True
            
        self.model = None
        self.report_function_dict = None
        self.model_cv = None
        
    def _override_config(self,config):
        """
        config should be a dictionary or yaml file.
        configuration handled in this class is a tree-like dictionary.
        override the default configuration with input config.
        """
        if config is None:
            return
        if isinstance(config, str):
            config = yaml.load(open(config))
            
        def override(a, b):
            # recursive function
            for k,d in b.items():
                
                if isinstance(d,dict):
                    if k in a.keys():
                        override(a[k],d)
                    else:
                        a[k] = d
                else:
                    a[k] = d
        
        override(self.config,config)
        
    def _add_kwargs_to_config(self,kwargs):
        """
        override configuration dictionary with keywarded arguments.
        find the leaf node in configuration tree which match the keyward.
        then override the value.
        """
        added_keywords = []
        def recursive_add(kwargs,config):
            # recursive function
            if not isinstance(config,dict):
                return 
            else:
                for k,d in config.items():
                    if k in kwargs.keys():
                        config[k] = kwargs[k]
                        added_keywords.append(k)
                    else:
                        recursive_add(kwargs,d)
                        
        recursive_add(kwargs, self.config)
        
        # any non-found keyword in default will be regarded as 
        # keyword for hBayesDM
        for keyword,value in kwargs.items():
            if keyword not in added_keywords:
                self.config['HBAYESDM'][keyword] = value
        
        
    def _add_latent_info_kwargs(self, dm_model, process_name, kwargs):
        """
        find the function for calculating the target latent process from implemented models
        please refer to the document for available models and latent processes
        """
        modelling_module = f'mbmvpa.preprocessing.computational_modeling.{dm_model}'
        modelling_module = importlib.import_module(modelling_module)
        kwargs['latent_function_dfwise'] = modelling_module.ComputationalModel(process_name)
        
        if process_name in modelling_module.latent_process_onset.keys():
            kwargs['onset_name'] = modelling_module.latent_process_onset[process_name]
    
    def _copy_config(self):
        """
        deep copy of configuration dictionary,
        skipping values which are not writable on yaml file.
        """
        def is_writable(d):
            if isinstance(d,str) or \
                isinstance(d, list) or \
                isinstance(d, tuple) or \
                isinstance(d, int) or \
                isinstance(d, float): 
                return True
            else:
                return False
                
        def recursive_copy(config):
            copied = {}
            for k,d in config.items():
                if isinstance(d,dict):
                    copied[k] = recursive_copy(d)
                elif is_writable(d):
                    copied[k] = d
            return(copied)
        
        return recursive_copy(self.config)
    
    def run(self,overwrite=False):
        """
        run the following procedures.
        
        1. preprocess fMRI & behavioral data
        2. load preprocessed data
        3. fit MVPA models and interprete the models
        """
        
        # X (fMRI): masking & zooming
        self.X_generator.run(overwrite=overwrite) 
        
        # y (latent process): comp. model. & hrf convolution
        self.y_generator.run(modeling_kwargs=self.config['HBAYESDM'],
                            overwrite=overwrite) 
        
        # set layout for loading X, y data
        self.bids_controller.reload()
        self.config['LOADER']['layout']=self.bids_controller.mbmvpa_layout
        self.loader = BIDSDataLoader(**self.config['LOADER'])
        
        # load X, y and voxel mask
        X_dict, y_dict = self.loader.get_data(subject_wise=True)
        voxel_mask = self.loader.get_voxel_mask()
        
        # set MVPA model and report function
        input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]
        self.config['MVPA'][self.mvpa_model_name]['input_shape'] = input_shape
        self.config['MVPA'][self.mvpa_model_name]['voxel_mask'] = voxel_mask
        self.model = self._mvpa_model_class(**self.config['MVPA'][self.mvpa_model_name])
        self.report_function_dict = self._mvpa_report_func(voxel_mask=voxel_mask, 
                                                          **self.config['MVPAREPORT'][self.mvpa_model_name])
        
        # set cross-validation module of MVPA (model_cv)
        self.model_cv = MVPA_CV(X_dict,
                                y_dict,
                                self.model,
                                report_function_dict=self.report_function_dict,
                                **self.config['MVPACV'])
        
        # run model_cv: fit models & interprete models 
        reports = self.model_cv.run()
        
        # save configuration
        save_config_path = str(self.model_cv.save_root / 'config.yaml')
        yaml.dump(self._copy_config(),open(save_config_path,'w'),indent=4, sort_keys=False)
        
        return reports