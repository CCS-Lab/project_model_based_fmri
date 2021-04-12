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


def run_mbmvpa(config=None,
              mvpa_model='elasticnet',
              report_path='.',
              overwrite=False,
              **kwargs):
    
    mbmvpa = MBMVPA(config=config,
                     mvpa_model=mvpa_model,
                     report_path=report_path,
                     **kwargs)

    return mbmvpa.run(overwrite=overwrite)
    

class MBMVPA():
    def __init__(self,
                 config=None,
                 mvpa_model='elasticnet',
                 report_path='.',
                 **kwargs):
        
        self.config = DEFAULT_ANALYSIS_CONFIGS
        self._override_config(config)
        self._add_kwargs_to_config(kwargs)
        
        if 'latent_function' not in self.config['LATENTPROCESS'].keys():
            self._add_latent_info_kwargs(self.config['LATENTPROCESS']['dm_model'],
                                         self.config['LATENTPROCESS']['process_name'], 
                                         self.config['LATENTPROCESS'])
            
        self.mvpa_model_name = mvpa_model
        result_name = '-'.join([''.join(self.config['LATENTPROCESS']['dm_model'].split('_')),
                                self.config['LOADER']['task_name'],
                                self.config['LOADER']['process_name'],
                                self.config['LOADER']['feature_name']])
        self.config['MVPAREPORT'][self.mvpa_model_name]['task_name'] = result_name
        self.config['MVPACV']['task_name'] = result_name
        
        
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
        
    def _override_config(self,config):

        if config is None:
            return
        if isinstance(config, str):
            config = yaml.load(open(config))
            
        def override(a, b):
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
        
        def recursive_add(kwargs,config):
            if not isinstance(config,dict):
                return 
            else:
                for k,d in config.items():
                    if k in kwargs.keys():
                        config[k] = kwargs[k]
                    else:
                        recursive_add(kwargs,d)
        
        recursive_add(kwargs, self.config)
        
            
    def _add_latent_info_kwargs(self, dm_model,process, kwargs):
        
        modelling_module = f'mbmvpa.preprocessing.computational_modeling.{dm_model}'
        modelling_module = importlib.import_module(modelling_module)
        latent_process_functions = modelling_module.latent_process_functions
        assert process in latent_process_functions.keys(), f"{process} func. is not defined."
        
        kwargs['modulation_dfwise'] = latent_process_functions[process]
        
        if process in modelling_module.latent_process_onset.keys():
            kwargs['onset_name'] = modelling_module.latent_process_onset[process]
    
    def _copy_config(self):
    
        def writable(d):
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
                elif writable(d):
                    copied[k] = d
            return(copied)
        
        return recursive_copy(self.config)
    
    def run(self,**kwargs):
        self.X_generator.run(**kwargs)
        self.y_generator.run(modelling_kwargs=self.config['HBAYESDM'],**kwargs)
        self.bids_controller.reload()
        self.config['LOADER']['layout']=self.bids_controller.mbmvpa_layout
        self.loader = BIDSDataLoader(**self.config['LOADER'])
        X_dict, y_dict = self.loader.get_data(subject_wise=True)
        voxel_mask = self.loader.get_voxel_mask()
        
        #if self.mvpa_model_name in NEED_INPUT_SHAPE_MODEL:
        input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]
        self.config['MVPA'][self.mvpa_model_name]['input_shape'] = input_shape
        #if self.mvpa_model_name in NEED_VOXELMASK_MODEL:
        self.model = self._mvpa_model_class(voxel_mask=voxel_mask,
                                            **self.config['MVPA'][self.mvpa_model_name])
        #else:
            #self.model = self._mvpa_model_class(**self.config['MVPA'][self.mvpa_model_name])
        self.report_function_dict = self._mvpa_report_func(voxel_mask=voxel_mask, 
                                                          **self.config['MVPAREPORT'][self.mvpa_model_name])
        
        self.model_cv = MVPA_CV(X_dict,
                                y_dict,
                                self.model,
                                report_function_dict=self.report_function_dict,
                                **self.config['MVPACV'])
        
        reports = self.model_cv.run()
        save_config_path = str(self.model_cv.save_root / 'config.yaml')
        yaml.dump(self._copy_config(),open(save_config_path,'w'),indent=4, sort_keys=False)
        
        return reports