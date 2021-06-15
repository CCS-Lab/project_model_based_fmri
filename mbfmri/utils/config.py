# default configuration for default names used in the package
ANAL_NAME = "MB-MVPA - Model based MVPA"
MBMVPA_PIPELINE_NAME = "MB-MVPA"
FMRIPREP_PIPELINE_NAME = "fMRIPrep"
TEMPLATE_SPACE = "MNI152NLin2009cAsym"
MAX_FMRIPREP_CHUNK_SIZE = 32
DEFAULT_DERIV_ROOT_DIR = "mbmvpa"
DEFAULT_ROI_MASK_DIR = "masks"
DEFAULT_VOXEL_MASK_FILENAME = "voxel_mask"
DEFAULT_FEATURE_SUFFIX = "voxelfeature"
DEFAULT_MODULATION_SUFFIX = "modulation"
DEFAULT_SIGNAL_SUFFIX = "signal"
DEFAULT_TIMEMASK_SUFFIX = "timemask"
DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME = "individual_params.tsv"
IGNORE_INDIV_PARAM = 'ignored'
DEFAULT_MODELCOMPARISON_FILENAME = 'model_comparison.tsv'
NIIEXT = 'nii.gz'
# default configuration for running MBMVPA
DEFAULT_ANALYSIS_CONFIGS = {
    'VOXELFEATURE': {
        'bids_layout':'.',
        'subjects':'all',
        'task_name':None,
        'fmriprep_name': 'fMRIPrep',
        'bold_suffix': 'bold',
        'confound_suffix': 'regressors',
        'mask_threshold': 2.58,
        'zoom': (2, 2, 2),
        'confounds': None,
        'smoothing_fwhm': 6,
        'standardize': True,
        'high_pass': 0.0078, # ~= 1/128
        'detrend': False,
        'n_thread': 1,
        'feature_name':'unnamed',
        'ignore_original':False,
        'space_name': None,
    },
    'LATENTPROCESS': {
        'bids_layout':'.',
        'subjects':'all',
        'task_name':None,
        'process_name':'unnamed',
        'computational_model':None,
        'dm_model':'unnamed',
        'hrf_model': 'glover',
        'use_duration': False,
        'n_core': 1,
        'onset_name': 'onset',
        'duration_name': 'duration',
        'end_name': None,
        'use_1sec_duration': True,
        'adjust_function': lambda v: v,
        'filter_function': lambda _: True,
        'adjust_function_dfwise': None,
        'filter_function_dfwise': None,
        'latent_function_dfwise': None,
        'latent_function': None,
        'computational_model': None,
        'skip_compmodel': False,
        'separate_run': False,
        'ignore_fmriprep': False,
        'criterion': 'looic',
        'lower_better': True,
    },
    'HBAYESDM': {
        },
    'LOADER':{
        'layout':'.',
        'subjects':'all',
        'reconstruct': False,
        'normalizer': 'none',
        'scale': [-1, 1],
        'verbose': 1,
        'task_name':None,
        'process_name':'unnamed',
        'feature_name':'unnamed',
        'use_absolute_value':False,
        'logistic':False,
        'binarizer_thresholds':None,
        'binarizer_ratios':.25,
    },
    'GLM':{
        'task_name': None,
        'process_name': None,
        'space_name': None,
        'smoothing_fwhm': 6,
        'mask_path': None,
        'mask_threshold': 2.58,
        'confounds': ['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
        'glm_save_path': '.',
        'hrf_model': 'glover',
        'drift_model': 'cosine',
        'high_pass': 1/128,
        'n_core': 4,
        'smoothing_fwhm': 6,
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
                'layer_dims': [1024, 1024],
                'activation': 'linear',
                'activation_output': 'linear',
                'dropout_rate': 0.5,
                'val_ratio': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'loss': 'mse',
                'n_epoch': 50,
                'n_patience': 10,
                'n_batch': 64,
                'n_sample': 100000,
                'use_bias': True,
                'gpu_visible_devices':[0],
                'logistic': False,
                'explainer': None,
                'train_verbosity': 0,
            },
            'cnn':{
                'layer_dims': [8, 16, 32],
                'kernel_size': [3, 3, 3],
                'logit_layer': 256,
                'activation': 'relu',
                'activation_output': 'linear',
                'dropout_rate': 0.2,
                'val_ratio': 0.2,
                'optimizer': 'adam',
                'loss': 'mse',
                'learning_rate': 0.001,
                'n_epoch': 50,
                'n_patience': 10,
                'n_batch': 64,
                'n_sample': 100000,
                'batch_norm': False,
                'gpu_visible_devices':[0],
                'logistic': False,
                'explainer': None,
                'train_verbosity': 0,
            },
        },
        'EXPLAINER':{
            'shap_explainer':'deep',
            'shap_null_background': False,
            'shap_n_background': 100,
            'shap_n_sample': 100,
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
                    'n_coef_plot': 150,
                    'map_type': 'z',
                    'sigma': 1,
                    'pval_threshold': 0.05
                },
                'mlp':{
                    'reports':['brainmap','pearsonr','mse'],
                    'map_type': 'z',
                    'sigma': 1,
                    'pval_threshold': 0.05
                },
                'cnn':{
                    'reports':['brainmap','pearsonr','mse'],
                    'map_type': 'z',
                    'sigma': 1,
                    'pval_threshold': 0.05
                },
            },
        'LOGISTICPOSTREPORT':{
                'elasticnet':{
                    'reports':['brainmap','accuracy','roc','elasticnet'],
                    'confidence_interval': 0.99,
                    'n_coef_plot': 150,
                    'map_type': 'z',
                    'sigma': 1,
                },
                'mlp':{
                    'reports':['brainmap','accuracy','roc'],
                    'map_type': 'z',
                    'sigma': 1,
                },
                'cnn':{
                    'reports':['brainmap','accuracy','roc'],
                    'map_type': 'z',
                    'sigma': 1,
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


# dict[hbayesdm model name] = list of pair(latent process, explanation) 
AVAILABLE_LATENT_PROCESS = {'bandit2arm_delta':[('EVchosen', 'expected value of chosen option'),
                                                ('PEchosen', 'prediction error of chosen option')],
                           }