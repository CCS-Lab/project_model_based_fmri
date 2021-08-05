#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.05.03

from mbfmri.preprocessing.events import LatentProcessGenerator
from mbfmri.preprocessing.bold import VoxelFeatureGenerator
from mbfmri.data.loader import BIDSDataLoader
from mbfmri.models.mvpa_general import MVPA_CV
from mbfmri.utils.report import PostReporter,FitReporter
from mbfmri.utils.explainer import Explainer
from mbfmri.core.base import MBFMRI
import mbfmri.utils.config
import yaml, importlib, copy

# dictionary for module path and class name for implemented model.
MVPA_MODEL_DICT = {'elasticnet':['mbfmri.models.elasticnet','MVPA_ElasticNet'],
                   'mlp':['mbfmri.models.tf_mlp','MVPA_MLP'],
                   'cnn':['mbfmri.models.tf_cnn','MVPA_CNN']}

NEED_RECONSTRUCT_MODEL = ['cnn']
USE_EXPLAINER = ['mlp','cnn']

def run_mbmvpa(config=None,
              mvpa_model='elasticnet',
              report_path='.',
              overwrite=False,
              overwrite_latent_process=True,
              refit_compmodel=False,
              **kwargs):
    
    r"""
    
    Callable function of the package to enable a single line usage.
    The following procedures are done by MBMVPA class.

    1. process fMRI & behavioral data to generate multi-voxel bold signals and latent process signals
    2. load processed signals.
    3. fit MVPA models and interprete the models to make a brain map.

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
    
    overwrite : bool, default=False
        Indicate if processing multi-voxel signals is required
        though the files exist.
    
    overwrite_latent : bool, default=False
        Indicate if generating latent process signals is required
        though the files exist.
        
    refit_compmodel : bool, default=False
        Indicate if fitting computational model is required
        though the fitted results (indiv. params. and LOOIC) exist.
        
    **kwargs : dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
        `Generating multi-voxel signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.bold>`_
        
        `Generating latent process signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.events>`_
        
        `Data loading document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.data.html#mbfmri.data.loader.BIDSDataLoader>`_
        
        `MVPA model document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.models.html>`_ (Please refer to the corresponding model according to *mvpa_model*.)
        
        Parameters of the above modules can be controlled by input paramter by keywords.
        (e.g. run_mbfmri(..., mask_smoothing_fwhm=6, ..., alpha=0.01) means mask_smoothing_fwhm will be set in VoxelFeatureGenerator and alpha will be set in ElasticNet.)
        
    Examples
    --------
    .. code:: python
    
        from mbfmri.core.mvpa import run_mbmvpa
        import hbayesdm

        _ = run_mbmvpa(analysis='mvpa',                     # name of analysis, "mvpa" or "glm"
                       bids_layout='mini_bornstein2017',    # data
                       mvpa_model='elasticnet',             # MVPA model, "mlp" or "cnn" for DNN
                       dm_model= 'banditNarm_lapse_decay',  # computational model
                       feature_name='zoom2rgrout',          # indentifier for processed fMRI data
                       task_name='multiarmedbandit',        # identifier for task
                       process_name='PEchosen',             # identifier for target latent process
                       subjects='all',                      # list of subjects to include
                       method='5-fold',                     # type of cross-validation
                       report_path=report_path,             # save path for reporting results
                       confounds=["trans_x", "trans_y",     # list of confounds to regress out
                                  "trans_z", "rot_x",
                                  "rot_y", "rot_z"],
                       n_core=4,                            # number of core for multi-processing in hBayesDM
                       n_thread=4,                          # number of thread for multi-threading in generating voxel features
                       overwrite=True,                      # indicate if re-run and overwriting are required
                       refit_compmodel=True,                # indicate if refitting comp. model is required
                      )
                      
    Full list of configuration
    --------------------------
    .. code:: python

        # located in mbfmri.utils.config.py
        'LATENTPROCESS': {
            'bids_layout':'.',
            'subjects':'all',
            'sessions': 'all',
            'save_path': None,
            'task_name':None,
            'process_name':'unnamed',
            'dm_model':'unnamed',
            'adjust_function': lambda v: v,
            'filter_function': lambda _: True,
            'latent_function': None,
            'adjust_function_dfwise': None,
            'filter_function_dfwise': None,
            'latent_function_dfwise': None,
            'individual_params': None,
            'skip_compmodel': False,
            'criterion': 'looic',
            'hrf_model': 'glover',
            'onset_name': 'onset',
            'duration_name': 'duration',
            'end_name': None,
            'use_1sec_duration': True,
            'mask_duration': False,
            't_r':None,
            'slice_time_ref':.5,
            'n_core': 1,
            'ignore_original': False,
            'ignore_fmriprep': False,
            'fmriprep_name': 'fMRIPrep',
            'event_suffix': 'events',
        },
        'VOXELFEATURE': {
            'bids_layout':'.',
            'subjects':'all',
            'sessions': 'all',
            'task_name':None,
            'feature_name':'unnamed',
            'fmriprep_name': 'fMRIPrep',
            'mask_path': None,
            'mask_threshold': 1.65,
            'mask_smoothing_fwhm':6,
            'include_default_mask': True,
            'gm_only': False,
            'atlas': None,
            'rois': [],
            'zoom': (2, 2, 2),
            'smoothing_fwhm': 6,
            'standardize': True,
            'high_pass': 0.0078, # ~= 1/128
            'detrend': True,
            'confounds': None,
            'n_thread': 1,
            'ignore_original':False,
            'space_name': None,
            't_r':None,
            'slice_time_ref':.5,
            'bold_suffix': 'bold',
            'confound_suffix': 'regressors',
        },
        'HBAYESDM': {
            },
        'LOADER':{
            'layout':'.',
            'subjects':'all',
            'sessions': 'all',
            'task_name':None,
            'process_name':'unnamed',
            'feature_name':'unnamed',
            'reconstruct': False,
            'y_normalizer': "none",
            'y_scale': [0,1],
            'y_std_threshold': 2.58,
            'y_clip': False,
            'y_use_absolute_value': False,
            'X_normalizer': "none",
            'X_scale': [0,1],
            'X_std_threshold': 2.58,
            'X_clip':False,
            'X_use_absolute_value': False,
            'logistic':False,
            'binarizer_positive_range': [.8,1.0],
            'binarizer_negative_range': [.0, .2],
            'binarizer_use_ratio':True,
            'verbose': 1,
        },
        'MVPA':{
            'MODEL':{
                'elasticnet':{
                    'alpha': 0.001,
                    'n_sample': 100000,
                    'max_lambda': 10,
                    'min_lambda_ratio': 1e-4,
                    'lambda_search_num': 100,
                    'n_jobs': 16,
                    'n_splits': 5,
                    'logistic': False,
                },
                'mlp':{
                    'layer_dims': [64, 64, 64],
                    'activation': 'sigmoid',
                    'activation_output': None,
                    'use_bias': True,
                    'dropout_rate': 0.5,
                    'batch_norm': False,
                    'logistic': False,
                    'l1_regularize':0,
                    'l2_regularize':0,
                    'val_ratio': 0.2,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'loss': None,
                    'n_batch': 64,
                    'n_epoch': 50,
                    'n_min_epoch': 0,
                    'n_patience': 10,
                    'n_sample': 100000,
                    'train_verbosity': 0,
                    'explainer': None,
                },
                'cnn':{
                    'layer_dims': [16, 16, 16],
                    'kernel_size': [3, 3, 3],
                    'logit_layer': 128,
                    'activation': 'relu',
                    'activation_output': None,
                    'dropout_rate': 0.2,
                    'batch_norm': True,
                    'logistic': False,
                    'l1_regularize':0,
                    'l2_regularize':0,
                    'val_ratio': 0.2,
                    'optimizer': 'adam',
                    'loss': None,
                    'learning_rate': 0.001,
                    'n_epoch': 50,
                    'n_min_epoch': 0,
                    'n_patience': 10,
                    'n_batch': 16,
                    'n_sample': 100000,
                    'train_verbosity': 0,
                    'explainer': None,
                },
            },
            'EXPLAINER':{
                'shap_explainer':'deep',
                'shap_n_background': 100,
                'shap_n_sample': 100,
                'pval_threshold': .05,
                'include_trainset':True,
                'voxel_mask': None,
            },
            'CV':{
                'method':'5-fold',
                'n_cv_repeat':1,
                'cv_save':True,
            },
            'POSTREPORT':{
                    'elasticnet':{
                        'reports':['brainmap','pearsonr','elasticnet','mse'],
                        'confidence_interval': 0.99,
                        'n_coef_plot': 'all',
                        'standardize': False,
                        'map_smoothing_fwhm': 0,
                        'map_threshold':0,
                        'cluster_threshold':0,
                        'pval_threshold': 0.05
                    },
                    'mlp':{
                        'reports':['brainmap','pearsonr','mse'],
                        'standardize': False,
                        'map_smoothing_fwhm': 0,
                        'map_threshold':0,
                        'cluster_threshold':0,
                        'pval_threshold': 0.05
                    },
                    'cnn':{
                        'reports':['brainmap','pearsonr','mse'],
                        'standardize': False,
                        'map_smoothing_fwhm': 0,
                        'map_threshold':0,
                        'cluster_threshold':0,
                        'pval_threshold': 0.05
                    },
                },
            'LOGISTICPOSTREPORT':{
                    'elasticnet':{
                        'reports':['brainmap','accuracy','roc','elasticnet'],
                        'confidence_interval': 0.99,
                        'n_coef_plot': 'all',
                        'standardize': False,
                        'map_smoothing_fwhm': 0,
                        'map_threshold':0,
                        'cluster_threshold':0,
                    },
                    'mlp':{
                        'reports':['brainmap','accuracy','roc'],
                        'standardize': False,
                        'map_smoothing_fwhm': 0,
                        'map_threshold':0,
                        'cluster_threshold':0,
                    },
                    'cnn':{
                        'reports':['brainmap','accuracy','roc'],
                        'standardize': False,
                        'map_smoothing_fwhm': 0,
                        'map_threshold':0,
                        'cluster_threshold':0,
                    },
                },
            'FITREPORT':{
                'metrics':['r','mse'],
            },
            'LOGISTICFITREPORT':{
                'metrics':['accuracy','auc'],
            },
        },
        'DATAPLOT':{
            '_height': 5,
            '_width': 10,
            '_fontsize': 12
        }
    }

    """
    
    mbmvpa = MBMVPA(config=config,
                     mvpa_model=mvpa_model,
                     report_path=report_path,
                     **kwargs)

    return mbmvpa.run(overwrite=overwrite,
                     overwrite_latent_process=overwrite_latent_process,
                     refit_compmodel=refit_compmodel)
    

class MBMVPA(MBFMRI):
    
    r"""
    
    Wrapper of functions in the package to enable a single line usage.
    The following procedures are done by MBMVPA class.

    1. process fMRI & behavioral data to generate multi-voxel bold signals and latent process signals
    2. load processed signals.
    3. fit MVPA models and interprete the models to make a brain map.

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
        
    **kwargs : dict
        Dictionary for keywarded arguments.
        This allows users to override default configuration and *config* input.
        Argument names are same as those of wrapped modules.
        
        `Generating multi-voxel signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.bold>`_
        
        `Generating latent process signals document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.preprocessing.html#mbfmri.preprocessing.events>`_
        
        `Data loading document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.data.html#mbfmri.data.loader.BIDSDataLoader>`_
        
        `MVPA model document <https://project-model-based-fmri.readthedocs.io/en/latest/mbfmri.models.html>`_ (Please refer to the corresponding model according to *mvpa_model*.)
        
        Parameters of the above modules can be controlled by input paramter by keywords.
        (e.g. run_mbfmri(..., mask_smoothing_fwhm=6, ..., alpha=0.01) means mask_smoothing_fwhm will be set in VoxelFeatureGenerator and alpha will be set in ElasticNet.)
         
    """
    def __init__(self,
                 config=None,
                 mvpa_model='elasticnet',
                 report_path='.',
                 logistic=False,
                 **kwargs):
        
        # load & set configuration
        self.config = self._load_default_config()
        self._override_config(config)
        kwargs['logistic']=logistic
        self._add_kwargs_to_config(kwargs)
        self.config['MVPA']['CV']['cv_save_path']=report_path
        
        # setting name for saving outputs
        # TODO - move setting 'result_name' while running.
        
        self.mvpa_model_name = mvpa_model
        
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
        self.logistic=logistic
        self.config['APPENDIX'] = {}
    
    def _set_result_name(self):
        dm_model_name = self.y_generator.best_model
        dm_model_name = ''.join(dm_model_name.split('_'))
        result_name = '-'.join([dm_model_name,
                                self.loader.task_name,
                                self.loader.process_name,
                                self.loader.feature_name])
        if self.logistic:
            result_name += '-logistic'
            self.config['MVPA']['LOGISTICPOSTREPORT']\
                [self.mvpa_model_name]['experiment_name'] = result_name
        else:
            self.config['MVPA']['POSTREPORT']\
                [self.mvpa_model_name]['experiment_name'] = result_name
            
        self.config['MVPA']['CV']['experiment_name'] = result_name
        
        
    def run(self,
            overwrite=False,
            overwrite_latent_process=True,
            refit_compmodel=False):
        """
        run the following procedures.
        
        1. preprocess fMRI & behavioral data
        2. load preprocessed data
        3. fit MVPA models and interprete the models
        
        Parameters
        ----------

        overwrite : bool, default=False
            Indicate if processing multi-voxel signals is required
            though the files exist.

        overwrite_latent : bool, default=False
            Indicate if generating latent process signals is required
            though the files exist.

        refit_compmodel : bool, default=False
            Indicate if fitting computational model is required
            though the fitted results (indiv. params. and LOOIC) exist.
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
        self._set_result_name()
        input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]
        self.config['MVPA']['MODEL'][self.mvpa_model_name]['input_shape'] = input_shape
        if self.mvpa_model_name in USE_EXPLAINER:
            self.config['MVPA']['EXPLAINER']['voxel_mask']=voxel_mask
            self.config['MVPA']['MODEL'][self.mvpa_model_name]['explainer'] = Explainer(**self.config['MVPA']['EXPLAINER'])
            
            
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
        self.model_cv = MVPA_CV(X_dict,y_dict,self.model,
                                post_reporter=self.post_reporter,
                                fit_reporter=self.fit_reporter,
                                **self.config['MVPA']['CV'])
        
        # run model_cv: fit models & interprete models 
        model_save_path = self.model_cv.save_root / 'best_model'
        outputs = self.model_cv.run(model_save_path=model_save_path)
        
        # save updated configuration
        self.save_config_path = str(self.model_cv.save_root / 'config.yaml')
        yaml.dump(self._copy_config(),open(self.save_config_path,'w'),indent=4, sort_keys=False)
        
        return outputs