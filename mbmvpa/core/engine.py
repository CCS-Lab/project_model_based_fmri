from ..preprocessing.events import LatentProcessGenerator
from ..preprocessing.bold import VoxelFeatureGenerator
from ..data.loader import BIDSDataLoader
from ..models.mvpa_general import MVPA_CV
from ..utils.config import DEFAULT_ANALYSIS_CONFIGS
from ..utils.report import build_elasticnet_report_functions, build_base_report_functions
import yaml, importlib, copy

MVPA_MODEL_DICT = {'elasticnet':['mbmvpa.models.elasticnet','MVPA_ElasticNet'],
                   'mlp':['mbmvpa.models.tf_mlp','MVPA_MLP'],
                   'cnn':['mbmvpa.models.tf_cnn','MVPA_CNN']}

MVPA_REPORT_DICT = {'elasticnet':build_elasticnet_report_functions,
                   'mlp':build_base_report_functions,
                   'cnn':build_base_report_functions}

NEED_INPUT_SHAPE_MODEL = ['mlp','cnn']
NEED_RECONSTRUCT_MODEL = ['cnn']
NEED_VOXELMASK_MODEL = ['cnn']

def run_mbmvpa(root,
                 dm_model,
                 process,
                 task=None,
                 feature="unnamed",
                 config=None,
                 mvpa_model='elasticnet',
                 subjects=None,
                 save_path=None,
                 mask_path=None,
                 report_path='.',
                 overwrite=False,
                 **kwargs):
    
    mbmvpa = MBMVPA(root,
                 dm_model,
                 process,
                 task,
                 feature,
                 config,
                 mvpa_model,
                 subjects,
                 save_path,
                 mask_path,
                 report_path,
                 **kwargs)
    
    return mbmvpa.run(overwrite=overwrite)
    

class MBMVPA():
    def __init__(self,
                 root,
                 dm_model,
                 process,
                 task=None,
                 feature="unnamed",
                 config=None,
                 mvpa_model='elasticnet',
                 subjects=None,
                 save_path=None,
                 mask_path=None,
                 report_path='.',
                 **kwargs):
        
        if isinstance(config, str):
            self.config = yaml.load(open(config))
        elif config is None:
            self.config = DEFAULT_ANALYSIS_CONFIGS
            
        if 'latent_function' not in self.config['LATENTPROCESS'].keys():
            self._add_latent_info_kwargs(dm_model, process, self.config['LATENTPROCESS'])
            
        self.config['LATENTPROCESS']['bids_layout'] = root
        self.config['LATENTPROCESS']['task_name'] = task
        self.config['LATENTPROCESS']['process_name'] = process
        self.config['LATENTPROCESS']['dm_model'] = dm_model
        self.config['VOXELFEATURE']['bids_layout'] = root
        self.config['VOXELFEATURE']['task_name'] = task
        self.config['VOXELFEATURE']['feature_name'] = feature
        if save_path is not None:
            self.config['VOXELFEATURE']['save_path'] = save_path
            self.config['LATENTPROCESS']['save_path'] = save_path
        if mask_path is not None:
            self.config['VOXELFEATURE']['mask_path'] = mask_path
        self.config['LOADER']['task_name'] = task
        self.config['LOADER']['process_name'] = process
        self.config['LOADER']['feature_name'] = feature
        self.config['MVPACV']['cv_save_path'] = report_path
        
        self.mvpa_model_name = mvpa_model
        result_name = ''.join(dm_model.split('_'))+"-"+ task+"-"+process+"-"+feature
        self.config['MVPAREPORT'][self.mvpa_model_name]['task_name'] = result_name
        self.config['MVPACV']['task_name'] = result_name
        
        self._add_kwargs_to_config(kwargs)
        
        self.X_generator = VoxelFeatureGenerator(**self.config['VOXELFEATURE'])
        self.bids_controller = self.X_generator.bids_controller
        self.y_generator = LatentProcessGenerator(bids_controller=self.bids_controller,
                                                  **self.config['LATENTPROCESS'])
        self.loader = None
        self._mvpa_model_class = getattr(
                                    importlib.import_module(MVPA_MODEL_DICT[mvpa_model][0]),
                                    MVPA_MODEL_DICT[mvpa_model][1])
        self._mvpa_report_func = MVPA_REPORT_DICT[mvpa_model]
        
        
        if self.mvpa_model_name in NEED_RECONSTRUCT_MODEL:
            self.config['LOADER']['reconstruct'] = True
        self.model = None
        self.report_function_dict = None
        self.model_cv = None
        
    
    def _add_kwargs_to_config(self,kwargs):
        
        def recursive(kwargs,config):
            if not isinstance(config,dict):
                return 
            else:
                for k,d in config.items():
                    if k in kwargs.keys():
                        config[k] = kwargs[k]
                    else:
                        recursive(kwargs,d)
        
        recursive(kwargs, self.config)
        
            
    def _add_latent_info_kwargs(self, dm_model,process, kwargs):
        
        modelling_module = f'mbmvpa.preprocessing.computational_modeling.{dm_model}'
        modelling_module = importlib.import_module(modelling_module)
        latent_process_functions = modelling_module.latent_process_functions
        assert process in latent_process_functions.keys(), f"{proces} func. is not defined."
        
        kwargs['modulation_dfwise'] = latent_process_functions[process]
        
        if process in modelling_module.latent_process_onset.keys():
            kwargs['onset_name'] = modelling_module.latent_process_onset[process]
        
    def run(self,**kwargs):
        self.X_generator.run(**kwargs)
        self.y_generator.run(modelling_kwargs=self.config['HBAYESDM'],**kwargs)
        self.bids_controller.reload()
        self.loader = BIDSDataLoader(layout=self.bids_controller.mbmvpa_layout, 
                                    **self.config['LOADER'])
        X_dict, y_dict = self.loader.get_data(subject_wise=True)
        voxel_mask = self.loader.get_voxel_mask()
        
        if self.mvpa_model_name in NEED_INPUT_SHAPE_MODEL:
            input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]
            self.config['MVPA'][self.mvpa_model_name]['input_shape'] = input_shape
        if self.mvpa_model_name in NEED_VOXELMASK_MODEL:
            self.model = self._mvpa_model_class(voxel_mask=voxel_mask,
                                                **self.config['MVPA'][self.mvpa_model_name])
        else:
            self.model = self._mvpa_model_class(**self.config['MVPA'][self.mvpa_model_name])
        self.report_function_dict = self._mvpa_report_func(voxel_mask=voxel_mask, 
                                                          **self.config['MVPAREPORT'][self.mvpa_model_name])
        
        self.model_cv = MVPA_CV(X_dict,
                                y_dict,
                                self.model,
                                report_function_dict=self.report_function_dict,
                                **self.config['MVPACV'])
        
        reports = self.model_cv.run()
        save_config_path = str(self.model_cv.save_root / 'config.yaml')
        config = copy.deepcopy(self.config)
        config['LATENTPROCESS']['adjust_function'] = None
        config['LATENTPROCESS']['filter_function'] = None
        config['LATENTPROCESS']['modulation_dfwise'] = None
        yaml.dump(config,open(save_config_path,'w'),indent=4, sort_keys=False)
        
        return reports