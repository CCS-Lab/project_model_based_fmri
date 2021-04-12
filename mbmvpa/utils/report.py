from mbmvpa.models.elasticnet import plot_elasticnet_result
from .coef2map import get_map
from .plot import plot_pearsonr
from scipy.stats import pearsonr
import numpy as np

class Report_BrainMap():
    
    def __init__(self,
                voxel_mask,
             task_name='unnamed',
             map_type='z',
             sigma=1):
        
        self.voxel_mask = voxel_mask
        self.task_name = task_name
        self.map_type = map_type
        self.sigma = sigma
        
    def __call__(self,
                 save,
                 save_path,
                 weights
                ):
        
        if isinstance(weights, dict):
            coefs = np.array([np.squeeze(data) for _, data in weights.items()]) 
        else:
            coefs = weights.reshape(-1, weights.shape[-1])
            
        get_map(coefs, self.voxel_mask, self.task_name,
                map_type=self.map_type, save_path=save_path, sigma=self.sigma)

class Report_ElasticNet():
    
    def __init__(self,
                confidence_interval=.99,
                 n_coef_plot=150):
        
        self.confidence_interval = confidence_interval
        self.n_coef_plot = n_coef_plot
    
    def __call__(self,
                 save,
                 save_path,
                 cv_mean_score,
                 cv_standard_error,
                 lambda_path,
                 lambda_best,
                 coef_path):
        
        plot_elasticnet_result(save_root=save_path, 
                           save=save,
                           cv_mean_score=cv_mean_score, 
                           cv_standard_error=cv_standard_error,
                           lambda_path=lambda_path,
                           lambda_val=lambda_best,
                           coef_path=coef_path,
                           confidence_interval=self.confidence_interval,
                           n_coef_plot=self.n_coef_plot)
        
class Report_PearsonR():
    def __init__(self, pval_threshold=0.01):
        self.pval_threshold = pval_threshold
        
    def __call__(self,
                 save,
                 save_path,
                 y_train,
                 y_test,
                 pred_train,
                 pred_test):
        
        key_list = list(y_train.keys())
        
        plot_pearsonr(save=save,
                     save_path=save_path,
                        y_train=[y_train[key] for key in key_list],
                        y_test=[y_test[key] for key in key_list],
                        pred_train=[pred_train[key] for key in key_list],
                        pred_test=[pred_test[key] for key in key_list],
                     pval_threshold=self.pval_threshold)
        
        
def build_base_report_functions(voxel_mask,
                             task_name='unnamed',
                             map_type='z',
                             sigma=1
                             ):
    
    report_function_dict = {}

    report_function_dict[('brain_map',
                          'weights') ] = Report_BrainMap(voxel_mask,
                                                         task_name,
                                                         map_type,
                                                         sigma)
    report_function_dict[('pearsonr',
                         'y_train',
                         'y_test',
                         'pred_train',
                         'pred_test')] = Report_PearsonR()
    
    return report_function_dict


def build_elasticnet_report_functions(voxel_mask,
                                     confidence_interval=.99,
                                     n_coef_plot=150,
                                     task_name='unnamed',
                                     map_type='z',
                                     sigma=1
                                     ):
    
    report_function_dict = {}
        
    report_function_dict[('elasticnet_report',
                          'cv_mean_score',
                          'coef_path',
                          'cv_standard_error',
                          'lambda_best',
                          'lambda_path')] = Report_ElasticNet(confidence_interval,n_coef_plot)

    report_function_dict[('brain_map',
                          'weights') ] = Report_BrainMap(voxel_mask,
                                                         task_name,
                                                         map_type,
                                                         sigma)
    report_function_dict[('pearsonr',
                         'y_train',
                         'y_test',
                         'pred_train',
                         'pred_test')] = Report_PearsonR()
    
    return report_function_dict



