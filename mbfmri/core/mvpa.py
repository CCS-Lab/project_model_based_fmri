#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.05.03

from mbfmri.preprocessing.events import LatentProcessGenerator
from mbfmri.preprocessing.bold import VoxelFeatureGenerator
from mbfmri.data.loader import BIDSDataLoader
from mbfmri.models.mvpa_general import MVPA_CV, MVPA_CV_H
from mbfmri.utils.report import PostReporter,FitReporter
from mbfmri.core.base import MBFMRI
import mbfmri.utils.config
import yaml, importlib, copy

# dictionary for module path and class name for implemented model.
MVPA_MODEL_DICT = {'elasticnet':['mbfmri.models.elasticnet','MVPA_ElasticNet'],
                   'mlp':['mbfmri.models.tf_mlp','MVPA_MLP'],
                   'cnn':['mbfmri.models.tf_cnn','MVPA_CNN']}

NEED_RECONSTRUCT_MODEL = ['cnn']

def run_mbmvpa(config=None,
              mvpa_model='elasticnet',
              report_path='.',
              overwrite=False,
              overwrite_latent_process=True,
              refit_compmodel=False,
              level=None,
              **kwargs):
    
    # callable wrapper of MBMVPA
    
    r"""
    
    Callable function of MB-MVPA package to enable a single line usage.
    The following procedures are done by MBMVPA class.

    1. preprocess fMRI & behavioral data
    2. load preprocessed data
    3. fit MVPA models and interprete the models

    By running this code, users can expect to get a brain activation pattern attributed to
    the target latent process defined in the computational model. 
    
    Parameters
    ----------
    
    config : dict or str or pathlib.PosixPath, default=None
        Dictionary for keyworded configuration, or path for yaml file.
        The configuration input will override the default configuration.
    mvpa_model : str, default="elasticnet"
        Name for MVPA model. Currently, "elasticnet," "mlp" and "cnn" are allowed.
    report_path : str or pathlib.PosixPath, defualt="."
        Path for saving outputs of MVPA_CV module. 
        please refer to mbmvpa.models.mvpa_general.MVPA_CV
    level : str, defualt=None
        if 'hierarchical' or 'H', use MVPA_CV_1stL class instead to run hiearchical version.
        The hiearchical version of the MB-MVPA analysis is composed of two parts.
        1) Run individual MB-MVPA on each subject
        2) creat (one sample) T-map using brain maps from each subject.
    **kwargs : dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
    """
    
    mbmvpa = MBMVPA(config=config,
                     mvpa_model=mvpa_model,
                     report_path=report_path,
                     level=level,
                     **kwargs)

    return mbmvpa.run(overwrite=overwrite,
                     overwrite_latent_process=overwrite_latent_process,
                     refit_compmodel=refit_compmodel)
    

class MBMVPA(MBFMRI):
    
    r"""
    
    Wrapper of functions in MB-MVPA package to enable a single line usage.
    The following procedures are done by MBMVPA class.

    1. preprocess fMRI & behavioral data
    2. load preprocessed data
    3. fit MVPA models and interprete the models

    By running this code, users can expect to get a brain activation pattern attributed to
    the target latent process defined in the computational model. 
    
    Parameters
    ----------
    
    config : dict or str or pathlib.PosixPath, default=None
        Dictionary for keyworded configuration, or path for yaml file.
        The configuration input will override the default configuration.
    mvpa_model : str, default="elasticnet"
        Name for MVPA model. Currently, "elasticnet," "mlp" and "cnn" are allowed.
    report_path : str or pathlib.PosixPath, defualt="."
        Path for saving outputs of MVPA_CV module. 
        please refer to mbmvpa.models.mvpa_general.MVPA_CV
    level : str, defualt=None
        if 'hierarchical' or 'H', use MVPA_CV_H class instead to run hiearchical version.
        The hiearchical version of the MB-MVPA analysis is composed of two parts.
        1) Run individual MB-MVPA on each subject
        2) creat (one sample) T-map using brain maps from each subject.
    **kwargs : dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
    """
    def __init__(self,
                 config=None,
                 mvpa_model='elasticnet',
                 report_path='.',
                 logistic=False,
                 level=None,
                 **kwargs):
        
        # load & set configuration
        self.config = self._load_default_config()
        self._override_config(config)
        kwargs['logistic']=logistic
        self._add_kwargs_to_config(kwargs)
        self.config['MVPA']['CV']['cv_save_path']=report_path
        
        # setting name for saving outputs
        self.mvpa_model_name = mvpa_model
        dm_model_name = self.config['LATENTPROCESS']['dm_model']
        if isinstance(dm_model_name,str):
            dm_model_name = ''.join(dm_model_name.split('_'))
        elif isinstance(dm_model_name,list) or \
            isinstance(dm_model_name,tuple):
            if len(dm_model_name) ==1 :
                dm_model_name = ''.join(dm_model_name[0].split('_'))
            else:
                dm_model_name = 'modelcomparison'
            
        result_name = '-'.join([dm_model_name,
                                self.config['LOADER']['task_name'],
                                self.config['LOADER']['process_name'],
                                self.config['LOADER']['feature_name']])
        self.config['MVPA']['POSTREPORT'][self.mvpa_model_name]['experiment_name'] = result_name
        self.config['MVPA']['CV']['experiment_name'] = result_name
        
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
        
        # set flag if reconstructing voxel feature (1D) to 4D is required
        if self.mvpa_model_name in NEED_RECONSTRUCT_MODEL:
            self.config['LOADER']['reconstruct'] = True
            
        self.model = None
        self.reporter = None
        self.model_cv = None
        if level is None:
            self.model_cv_builder = MVPA_CV
        elif level.lower() in ['hierarchical','h']:
            self.model_cv_builder = MVPA_CV_H
        
        self.config['APPENDIX'] = {}
        self.logistic=logistic
    
    def run(self,
            overwrite=False,
            overwrite_latent_process=True,
            refit_compmodel=False):
        """
        run the following procedures.
        
        1. preprocess fMRI & behavioral data
        2. load preprocessed data
        3. fit MVPA models and interprete the models
        """
        
        # X (fMRI): masking & zooming
        self.X_generator.run(overwrite=overwrite) 
        
        # y (latent process): comp. model. & hrf convolution
        self.config['HBAYESDM']['refit_compmodel']=refit_compmodel
        self.y_generator.run(modeling_kwargs=self.config['HBAYESDM'],
                            overwrite=overwrite|overwrite_latent_process) 
        self.config['APPENDIX']['best_model'] = self.y_generator.best_model
        # reload bids layout and plot processed data
        self.bids_controller.reload()
        self.bids_controller.plot_processed_data(feature_name=self.X_generator.feature_name,
                                                 process_name=self.y_generator.process_name,
                                                h=self.config['DATAPLOT']['_height'],
                                                w=self.config['DATAPLOT']['_width'],
                                                fontsize=self.config['DATAPLOT']['_fontsize'])
        # set layout for loading X, y data
        self.config['LOADER']['layout']=self.bids_controller.mbmvpa_layout
        self.loader = BIDSDataLoader(**self.config['LOADER'])
        
        # load X, y and voxel mask
        X_dict, y_dict = self.loader.get_data(subject_wise=True)
        voxel_mask = self.loader.get_voxel_mask()
        
        # set MVPA model and report function
        input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]
        self.config['MVPA']['MODEL'][self.mvpa_model_name]['input_shape'] = input_shape
        self.config['MVPA']['MODEL'][self.mvpa_model_name]['voxel_mask'] = voxel_mask
        self.model = self._mvpa_model_class(**self.config['MVPA']['MODEL'][self.mvpa_model_name])
        if self.logistic:
            self.fit_reporter = FitReporter(**self.config['MVPA']['LOGISTICFITREPORT'])
            self.post_reporter = PostReporter(voxel_mask=voxel_mask,
                                 **self.config['MVPA']['LOGISTICPOSTREPORT'][self.mvpa_model_name])
        else:
            self.fit_reporter = FitReporter(**self.config['MVPA']['FITREPORT'])
            self.post_reporter = PostReporter(voxel_mask=voxel_mask,
                                 **self.config['MVPA']['POSTREPORT'][self.mvpa_model_name])
        
        # set cross-validation module of MVPA (model_cv)
        self.model_cv = self.model_cv_builder(X_dict,
                                                y_dict,
                                                self.model,
                                                post_reporter=self.post_reporter,
                                                fit_reporter=self.fit_reporter,
                                                **self.config['MVPA']['CV'])
        
        # run model_cv: fit models & interprete models 
        outputs = self.model_cv.run()
        
        # save configuration
        save_config_path = str(self.model_cv.save_root / 'config.yaml')
        yaml.dump(self._copy_config(),open(save_config_path,'w'),indent=4, sort_keys=False)
        
        return outputs