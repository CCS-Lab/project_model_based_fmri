# Fixed name
ANAL_NAME = "MB-MVPA - Model based MVPA"
MBMVPA_PIPELINE_NAME = "MB-MVPA"
FMRIPREP_PIPELINE_NAME = "fMRIPrep"
TEMPLATE_SPACE = "MNI152NLin2009cAsym"
MAX_FMRIPREP_CHUNK_SIZE = 32
# default configuration for default names used in the package
DEFAULT_DERIV_ROOT_DIR = "mbmvpa"
DEFAULT_ROI_MASK_DIR = "masks"
DEFAULT_VOXEL_MASK_FILENAME = "voxel_mask"
DEFAULT_FEATURE_SUFFIX = "voxelfeature"
DEFAULT_MODULATION_SUFFIX = "modulation"
DEFAULT_SIGNAL_SUFFIX = "signal"
DEFAULT_TIMEMASK_SUFFIX = "timemask"
DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME = "individual_params.tsv"
DEFAULT_SAVE_PATH_CKPT = "mvpa/fitting_result"
# configuration
DEFAULT_ANALYSIS_CONFIGS = {
    'VOXELFEATURE': {
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
        'n_thread': 4,
    },
    'LATENTPROCESS': {
        'hrf_model': 'glover',
        'use_duration': False,
        'n_core': 4,
        'onset_name': 'onset',
        'duration_name': 'duration',
        'end_name': None,
        'use_1sec_duration': True,
    },
    'HBAYESDM': {
        },
    'LOADER':{
        'reconstruct': False,
        'normalizer': 'none',
        'scale': [-1, 1],
        'verbose': 1,
    },
    'MVPA':{
        'elasticnet':{
            'alpha': 0.001,
            'n_samples': 100000,
            'shuffle': True,
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
        }
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
            'sigma': 1
        },
        'cnn':{
            'map_type': 'z',
            'sigma': 1,
        },
    },
}