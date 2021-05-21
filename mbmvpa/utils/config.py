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
    },
    'LATENTPROCESS': {
        'bids_layout':'.',
        'subjects':'all',
        'task_name':None,
        'process_name':'unnamed',
        'computational_model':None,
        'dm_model':None,
        'hrf_model': 'glover',
        'use_duration': False,
        'n_core': 1,
        'onset_name': 'onset',
        'duration_name': 'duration',
        'end_name': None,
        'use_1sec_duration': True,
        'adjust_function': lambda v: v,
        'filter_function': lambda _: True,
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
    },
    'MVPA':{
        'elasticnet':{
            'alpha': 0.001,
            'n_sample': 100000,
            'max_lambda': 10,
            'min_lambda_ratio': 1e-4,
            'lambda_search_num': 100,
            'n_jobs': 16,
            'n_splits': 5,
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
            'use_bipolar_balancing': False,
            'gpu_visible_devices':[0]
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
            'batch_norm': True,
            'use_bipolar_balancing': False,
            'gpu_visible_devices':[0]
        },
        'mlp_shap':{
            'layer_dims': [512, 512],
            'activation': 'linear',
            'activation_output': 'linear',
            'dropout_rate': 0.5,
            'val_ratio': 0.2,
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'loss': 'mse',
            'n_epoch': 50,
            'n_patience': 10,
            'n_batch': 32,
            'n_sample': 100000,
            'use_bias': True,
            'use_bipolar_balancing': False,
            'gpu_visible_devices':[0],
            'use_null_background': True,
            'background_num': 1000,
            'sample_num' : 100,
        },
    },
    'MVPACV':{
        'method':'5-fold',
        'n_cv_repeat':1,
        'cv_save':True,
        
    },
    'MVPAREPORT':{
        'elasticnet':{
            'confidence_interval': 0.99,
            'n_coef_plot': 150,
            'map_type': 'z',
            'sigma': 1
        },
        'mlp':{
            'map_type': 'z',
            'sigma': 1,
        },
        'cnn':{
            'map_type': 'z',
            'sigma': 1,
        },
        'mlp_shap':{
            'map_type': 'z',
            'sigma': 1,
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