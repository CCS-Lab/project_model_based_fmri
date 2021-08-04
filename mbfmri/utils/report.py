# TODO add reporting configuration

from mbfmri.utils.coef2map import get_map
from mbfmri.utils.plot import plot_pearsonr
from mbfmri.utils.report_utils import *
from scipy.stats import pearsonr
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import norm
import functools

report_function_dict = {'brainmap':{'module':Report_BrainMap,
                                   'data':['weights'],
                                   'parameter':['voxel_mask',
                                                'experiment_name',
                                                'standardize',
                                                'map_smoothing_fwhm',
                                                'map_threshold',
                                                'cluster_threshold'],
                                   'default':{'experiment_name':'unnamed',
                                              'standardize':True,
                                              'map_smoothing_fwhm':6,
                                              'map_threshold':0,
                                              'cluster_threshold':0
                                             }
                                   },
                        'r':{'module':Plot_R,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':['pval_threshold'],
                                    'default':{'pval_threshold':.01}
                                   },
                        'pearsonr':{'module':Plot_PearsonR,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':['pval_threshold'],
                                    'default':{'pval_threshold':.01}
                                   },
                        'spearmanr':{'module':Plot_SpearmanR,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':['pval_threshold'],
                                    'default':{'pval_threshold':.01}
                                   },
                        'mse':{'module':Plot_MSE,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':[],
                                    'default':{}
                                   },
                        'accuracy':{'module':Plot_Accuracy,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':[],
                                    'default':{}
                                   },
                        'roc':{'module':Plot_ROC,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':[],
                                    'default':{}
                                   },
                        'elasticnet':{'module':Plot_ElasticNet,
                                     'data':['cv_mean_score',
                                             'cv_standard_error',
                                             'lambda_path',
                                             'lambda_best',
                                             'coef_path'],
                                      'parameter':['confidence_interval',
                                                   'n_coef_plot'],
                                      'default':{'confidence_interval':.99,
                                                'n_coef_plot':'all'
                                                },
                                     },
                        }

metric_function_dict = {'mse':{'module':Metric_MSE,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':[],
                                    'default':{}
                                   },
                        
                        'r':{'module':Metric_R,
                               'data':['y_train',
                                       'y_test',
                                       'pred_train',
                                       'pred_test'],
                                'parameter':[],
                                'default':{}
                               },
                        'pearsonr':{'module':Metric_PearsonR,
                                   'data':['y_train',
                                           'y_test',
                                           'pred_train',
                                           'pred_test'],
                                    'parameter':[],
                                    'default':{}
                                   },
                        'spearmanr':{'module':Metric_SpearmanR,
                               'data':['y_train',
                                       'y_test',
                                       'pred_train',
                                       'pred_test'],
                                'parameter':[],
                                'default':{}
                               },
                        'auc':{'module':Metric_AUC,
                               'data':['y_train',
                                       'y_test',
                                       'pred_train',
                                       'pred_test'],
                                'parameter':[],
                                'default':{}
                               },
                        'accuracy':{'module':Metric_Accuracy,
                               'data':['y_train',
                                       'y_test',
                                       'pred_train',
                                       'pred_test'],
                                'parameter':[],
                                'default':{}
                               },
                        }


def aggregate(search_path,
             names,
             invalid_ids=[]):
    
    '''
    find files including input names
    this function is for navigating raw results from analysis
    '''
    search_path = Path(search_path)
    
    # make names as list of string
    if isinstance(names,str):
        names = [names]
        
    def checker(f):
        report_id = f.name.split('_')[0]
        return report_id not in invalid_ids
    
    data = {name:[f for f in search_path.glob(f'**/*{name}*') if checker(f)] for name in names }
    
    def cmp(v1,v2):
        fv1,bv1 = v1.name.split('_')[0].split('-')
        fv2,bv2 = v2.name.split('_')[0].split('-')
        fdiff = int(fv1)-int(fv2)
        bdiff = int(bv1)-int(bv2)
        return fdiff if fdiff !=0 else bdiff
    
        
    # sort
    for _, files in data.items():
        files.sort(key=functools.cmp_to_key(cmp))
    return data
    

class PostReporter():
    r"""
    PostReporter aggregates fitting results by name and makes reports. 
    Several helpful reports are implemented and users can input which
    kinds of reports should be made for their purposes.

    Parameters
    ----------

    reports : list of str, default=["brainmap"]
        List of name for reporting function. "brainmap", "r", "pearsonr", 
        "spearmanr", "mse", "accuracy", "roc", "elasticnet" are available

        Some of them require additional arguments like "voxel_mask" in "brainmap."
        Please refer to the document.

    """
    def __init__(self,
                reports=["brainmap"],
                **kwargs):
        
        self.function_dict = report_function_dict 
        self.reports = {report:self._init_report(report,**kwargs) for report in reports}
    
    def _init_report(self,
                    report,
                    **kwargs):

        report_module = self.function_dict[report]['module']
        parameter =  self.function_dict[report]['parameter']
        trimmed_kwargs = self.function_dict[report]['default']
        
        for p in parameter:
            if p in kwargs.keys():
                trimmed_kwargs[p]=kwargs[p]
                
            assert p in trimmed_kwargs.keys(), \
                f"ERROR: can't find report paramter-{p} for {report}"
            
        report_module = report_module(**trimmed_kwargs)
        
        return {'module': report_module,
                'data': self.function_dict[report]['data'],
               'invalid_ids':[]}
        
    def _load_data(self,
                   search_path,
                   names,
                   invalid_ids=[]):
        loaded = {}
        data = aggregate(search_path,names,invalid_ids)
        for name, files in data.items():
            loaded[name] = [np.load(f,allow_pickle=True) for f in files]
            
        return loaded
    
    def run(self,
            search_path='.',
            save=True,
            save_path='.'):
        outputs = {}
        for name, report in self.reports.items():
            report_kwargs = self._load_data(search_path,
                                            report['data'],
                                           report['invalid_ids'])
            report_save_path = Path(save_path)/name
            
            if save:
                report_save_path.mkdir(exist_ok=True)
            
            output = report['module'](save=save,
                              save_path=report_save_path,
                              **report_kwargs)
            if output is not None:
                outputs[name] = output
        report_names = list(self.reports.keys())
        print(f"INFO: report(s)-{report_names} is(are) done.")

        return outputs

class FitReporter():

    r"""
    FitReporter calculates metrics for the fitted result in each fold of 
    cross-validation.  
    Several metrics are implemented and users can input which
    kinds of reports should be made for their purposes.

    Parameters
    ----------

    metrics : list of str, default=[]
        List of name for metric function. "r", "pearsonr", 
        "spearmanr", "mse", "accuracy", "auc" are available

    """

    def __init__(self,
                metrics=[],
                **kwargs):
        self.function_dict = metric_function_dict
        self.reports = {report:self._init_report(report,**kwargs) for report in metrics}
    
    def _init_report(self,
                    report,
                    **kwargs):

        report_module = self.function_dict[report]['module']
        parameter =  self.function_dict[report]['parameter']
        trimmed_kwargs = self.function_dict[report]['default']
        
        for p in parameter:
            if p in kwargs.keys():
                trimmed_kwargs[p]=kwargs[p]
                
            assert p in trimmed_kwargs.keys(), \
                f"ERROR: can't find report paramter-{p} for {report}"
            
        report_module = report_module(**trimmed_kwargs)
        
        return {'module': report_module,
                'data': self.function_dict[report]['data']}
    
    def run(self,**kwargs):
        
        outputs = {}
        for name, report in self.reports.items():
            output = report['module'](**{d:kwargs[d] for d in report['data']})
            for output_key, output_value in output.items():
                outputs[name+'_'+output_key] = output_value
                
        return outputs
            
