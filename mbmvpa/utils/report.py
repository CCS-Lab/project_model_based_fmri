from .coef2map import get_map
from .plot import plot_pearsonr
from .report_utils import *
from scipy.stats import pearsonr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
import re

report_function_dict = {'brainmap':{'module':Report_BrainMap,
                                   'data':['weights'],
                                   'parameter':['voxel_mask',
                                                'experiment_name',
                                                'map_type',
                                                'sigma'],
                                   'default':{'experiment_name':'unnamed',
                                              'map_type':'z',
                                              'sigma':1,
                                             }
                                   },
                        'pearsonr':{'module':Report_PearsonR,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':['pval_threshold'],
                                    'default':{'pval_threshold':.01}
                                   },
                        'elasticnet':{'module':Report_ElasticNet,
                                     'data':['cv_mean_score',
                                             'cv_standard_error',
                                             'lambda_path',
                                             'lambda_best',
                                             'coef_path'],
                                      'parameter':['confidence_interval',
                                                   'n_coef_plot'],
                                      'default':{'confidence_interval':.99,
                                                'n_coef_plot':150
                                                },
                                     }
                        }

def aggregate(search_path,
             names):
    
    '''
    find files including input names
    this function is for navigating raw results from analysis
    '''
    search_path = Path(search_path)
    
    # make names as list of string
    if isinstance(names,str):
        names = [names]
        
    data = {name:[f for f in search_path.glob(f'**/*{name}*')] for name in names}
    
    # sort
    for _, files in data.items():
        files.sort(key=lambda v :int(re.sub("[^0-9]", "", v.stem)))
    return data
    

class Reporter():
    
    def __init__(self,
                reports=['brainmap','pearsonr'],
                **kwargs):
        self.reports = {report:self._init_report(report,**kwargs) for report in reports}
    
    def _init_report(self,
                    report,
                    **kwargs):
        
        report_module = report_function_dict[report]['module']
        parameter =  report_function_dict[report]['parameter']
        trimmed_kwargs = report_function_dict[report]['default']
        
        for p in parameter:
            if p in kwargs.keys():
                trimmed_kwargs[p]=kwargs[p]
                
            assert p in trimmed_kwargs.keys(), \
                f"ERROR: can't find report paramter-{p} for {report}"
            
        report_module = report_module(**trimmed_kwargs)
        
        return {'module': report_module,
                'data': report_function_dict[report]['data']}
        
    def _load_data(self,
                   search_path,
                   names):
        loaded = {}
        data = aggregate(search_path,names)
        for name, files in data.items():
            loaded[name] = np.array([np.load(f,allow_pickle=True) for f in files])
            
        return loaded
    
    def run(self,
            search_path='.',
            save=True,
            save_path='.'):
        
        for name, report in self.reports.items():
            report_kwargs = self._load_data(search_path,
                                            report['data'])
            report_save_path = Path(save_path)/name
            
            if save:
                report_save_path.mkdir(exist_ok=True)
            
            report['module'](save=save,
                              save_path=report_save_path,
                              **report_kwargs)
            
        report_names = list(self.reports.keys())
        print(f"INFO: report(s)-{report_names} is(are) done.")



