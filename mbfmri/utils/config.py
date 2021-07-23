# default configuration for default names used in the package
# TODO change names
ANAL_NAME = "MB-MVPA - Model based MVPA"
MBMVPA_PIPELINE_NAME = "MB-MVPA"
FMRIPREP_PIPELINE_NAME = "fMRIPrep"
TEMPLATE_SPACE = "MNI152NLin2009cAsym"
MAX_FMRIPREP_CHUNK_SIZE = 32
DEFAULT_DERIV_ROOT_DIR = "mbmvpa"
DEFAULT_ROI_MASK_DIR = "masks"
DEFAULT_MASK_EXCLUDE_DIR = "exclude"
DEFAULT_MASK_INCLUDE_DIR = "include"
DEFAULT_VOXEL_MASK_FILENAME = "voxel_mask"
DEFAULT_FEATURE_SUFFIX = "voxelfeature"
DEFAULT_MODULATION_SUFFIX = "modulation"
DEFAULT_SIGNAL_SUFFIX = "signal"
DEFAULT_TIMEMASK_SUFFIX = "timemask"
DEFAULT_INDIVIDUAL_PARAMETERS_FILENAME = "individual_params.tsv"
IGNORE_INDIV_PARAM = 'ignored'
DEFAULT_MODELCOMPARISON_FILENAME = 'model_comparison.tsv'
CONFOUNDEXT='tsv'
NIIEXT = 'nii.gz'
MODULATIONEXT ='tsv'
SPECEXT='json'
EVENTEXT='tsv'
VOXELFTEXT='npy'
MODSIGEXT='npy'
TIMEMASKEXT='npy'
# default configuration for running MBMVPA
DEFAULT_ANALYSIS_CONFIGS = {
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
        'reconstruct': False,
        'y_normalizer': "none",
        'y_use_absolute_value': False,
        'y_scale': [0,1],
        'y_std_threshold': 2.58,
        'y_clip': False,
        'X_normalizer': "none",
        'X_use_absolute_value': False,
        'X_scale': [0,1],
        'X_std_threshold': 2.58,
        'X_clip':False,
        'verbose': 1,
        'task_name':None,
        'process_name':'unnamed',
        'feature_name':'unnamed',
        'logistic':False,
        'binarizer_positive_range': [.8,1.0],
        'binarizer_negative_range': [.0, .2],
        'binarizer_use_ratio':True,
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
                'dropout_rate': 0.5,
                'val_ratio': 0.2,
                'optimizer': 'adam',
                'learning_rate': 0.001,
                'loss': 'mse',
                'n_epoch': 50,
                'n_min_epoch': 0,
                'n_patience': 10,
                'n_batch': 64,
                'n_sample': 100000,
                'l1_regularize':0,
                'l2_regularize':0,
                'use_bias': True,
                'logistic': False,
                'explainer': None,
                'train_verbosity': 0,
                'batch_norm':False
            },
            'cnn':{
                'layer_dims': [16, 16, 16],
                'kernel_size': [3, 3, 3],
                'logit_layer': 128,
                'activation': 'relu',
                'activation_output': None,
                'dropout_rate': 0.2,
                'val_ratio': 0.2,
                'optimizer': 'adam',
                'loss': 'mse',
                'learning_rate': 0.001,
                'n_epoch': 50,
                'n_min_epoch': 0,
                'n_patience': 10,
                'n_batch': 16,
                'n_sample': 100000,
                'l1_regularize':0,
                'l2_regularize':0,
                'batch_norm': True,
                'logistic': False,
                'explainer': None,
                'train_verbosity': 0,
            },
        },
        'EXPLAINER':{
            'shap_explainer':'deep',
            'shap_n_background': 100,
            'shap_n_sample': 100,
            'pval_threshold': .05,
            'include_trainset':True,
            'voxel_mask': None,
            'shap_save': True,
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
                    'standardize': False,
                    'map_smoothing_fwhm': 0,
                    'pval_threshold': 0.05
                },
                'mlp':{
                    'reports':['brainmap','pearsonr','mse'],
                    'standardize': False,
                    'map_smoothing_fwhm': 0,
                    'pval_threshold': 0.05
                },
                'cnn':{
                    'reports':['brainmap','pearsonr','mse'],
                    'standardize': False,
                    'map_smoothing_fwhm': 0,
                    'pval_threshold': 0.05
                },
            },
        'LOGISTICPOSTREPORT':{
                'elasticnet':{
                    'reports':['brainmap','accuracy','roc','elasticnet'],
                    'confidence_interval': 0.99,
                    'n_coef_plot': 150,
                    'standardize': False,
                    'map_smoothing_fwhm': 0,
                },
                'mlp':{
                    'reports':['brainmap','accuracy','roc'],
                    'standardize': False,
                    'map_smoothing_fwhm': 0,
                },
                'cnn':{
                    'reports':['brainmap','accuracy','roc'],
                    'standardize': False,
                    'map_smoothing_fwhm': 0,
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
    },
    'GLM':{
        'task_name': None,
        'process_name': None,
        'space_name': None,
        'mask_path': None,
        'mask_threshold': 2.58,
        'mask_smoothing_fwhm':6,
        'include_default_mask': True,
        'atlas': None,
        'rois': [],
        'gm_only': False,
        'glm_save_path': '.',
        'n_core': 4,
        'bold_suffix': 'bold',
        'confound_suffix': 'regressors',
        'subjects': 'all',
        'sessions': 'all',
        'zoom': (1,1,1),
        'img_filters': None,
        'slice_time_ref': 0.,
        'hrf_model': 'glover',
        'drift_model': 'cosine',
        'high_pass': 1/128,
        'drift_order': 1,
        'fir_delays': [0],
        'min_onset': -24,
        'target_affine': None,
        'target_shape': None,
        'smoothing_fwhm': 6, 
        'standardize': True,
        'signal_scaling': 0, 
        'noise_model': 'ar1',
        'verbose': 0, 
        'n_jobs': 1,
        'confound_names':['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
        'slice_time_ref':.5,
    },
}


dict_list = [DEFAULT_ANALYSIS_CONFIGS['VOXELFEATURE'],
             DEFAULT_ANALYSIS_CONFIGS['LATENTPROCESS'],
             DEFAULT_ANALYSIS_CONFIGS['HBAYESDM'],
             DEFAULT_ANALYSIS_CONFIGS['LOADER'],
             DEFAULT_ANALYSIS_CONFIGS['GLM'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['MODEL']['elasticnet'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['MODEL']['mlp'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['MODEL']['cnn'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['EXPLAINER'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['CV'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['FITREPORT'],
             DEFAULT_ANALYSIS_CONFIGS['MVPA']['LOGISTICFITREPORT'],
             DEFAULT_ANALYSIS_CONFIGS['DATAPLOT']]

import argparse

parser = argparse.ArgumentParser()

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1',''):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def str2intlist(v):
    return [int(s) for s in v.split(",")]

def str2floatlist(v):
    return [float(s) for s in v.split(",")]

def str2list(v):
    return v.split(",")

argument_list = []
for _dict in dict_list:
    for k,d in _dict.items():
        if isinstance(d,bool):
                parser_type = str2bool
        else:
            parser_type = type(d)

        if isinstance(d,list):
            if len(d) != 0:
                if isinstance(d[0],int):
                    parser_type = str2intlist
                elif isinstance(d[0],float):
                    parser_type = str2floatlist
                else:
                    parser_type = str2list
        else:
            if d is None:
                parser_type = str
            nargs="?"
        if k not in argument_list:
            parser.add_argument(f'--{k}', type=parser_type, default=None, nargs=nargs)
            argument_list.append(k)
        else:
            continue
                
parser.add_argument(f'--mvpa_model', type=str, default='elasticnet')
parser.add_argument(f'--report_path', type=str, default='.')
parser.add_argument(f'--analysis', type=str, default='mvpa')
parser.add_argument(f'--refit_compmodel', type=str2bool, default=False, nargs='?', const='')
parser.add_argument(f'--overwrite', type=str2bool, default=False, nargs='?', const='')

# dict[hbayesdm model name] = list of pair(latent process, explanation) 
AVAILABLE_LATENT_PROCESS = {'bandit2arm_delta':[('EVchosen', 'expected value of chosen option'),
                                                ('PEchosen', 'prediction error of chosen option')],
                           }


