mbfmri.core package
===================

Top wrapping function to run general model-based fMRI analysis.

1. process fMRI & behavioral data to generate multi-voxel bold signals and latent process signals
2. load processed signals.

Then 

**MVPA approach**: fit MVPA models and interprete the models to make a brain map.
**GLM approach**:
        

mbfmri.core.engine.run\_mbfmri
---------------------------------------------------------------

.. automodule:: mbfmri.core.engine
   :members:
   :undoc-members:
   :show-inheritance:



.. toctree::
   :maxdepth: 1
   :glob:
   :caption: By approach

   mbfmri.core.glm.rst
   mbfmri.core.mvpa.rst
   
   
Full list of configuration
--------------------------
.. code-block:: json

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
    },
    'GLM':{
        'task_name': None,
        'process_name': None,
        'space_name': None,
        'mask_path': None,
        'mask_threshold': 1.65,
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
        'min_onset': 0,
        'target_affine': None,
        'target_shape': None,
        'smoothing_fwhm': 6, 
        'standardize': True,
        'signal_scaling': 0, 
        'noise_model': 'ar1',
        'verbose': 0, 
        'n_jobs': 1,
        'confounds':['trans_x','trans_y','trans_z','rot_x', 'rot_y', 'rot_z'],
        't_r':None,
        'slice_time_ref':.5,
    },
}