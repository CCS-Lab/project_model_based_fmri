from ..models.elasticnet import plot_elasticnet_result
from ..utils.coef2map import get_map
import numpy as np

def build_elasticnet_report_functions(voxel_mask,
                                     confidence_interval=.99,
                                     n_coef_plot=150,
                                     task_name='unnamed',
                                     map_type='z',
                                     sigma=1
                                     ):
    
    report_function_dict = {}
    
    def plot_elasticnet_reuslt_2(save,save_path,
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
                           confidence_interval=confidence_interval,
                           n_coef_plot=n_coef_plot)
        
    report_function_dict[('cv_mean_score',
                          'coef_path',
                          'cv_standard_error',
                          'lambda_best',
                          'lambda_path')] = plot_elasticnet_reuslt_2
    
    def get_map_2(save,save_path,weights):
        coefs = np.array([np.squeeze(data) for _, data in weights.items()])  
        get_map(coefs, voxel_mask, task_name,
                map_type=map_type, save_path=save_path, sigma=sigma)
        
    report_function_dict['weights'] = get_map_2
    
    # TODO add pearson_r report
    
    return report_function_dict


def build_mlp_report_functions(voxel_mask,
                             task_name='unnamed',
                             map_type='z',
                             sigma=1
                             ):
    
    report_function_dict = {}
    
   
    def get_map_2(save,save_path,weights):
        coefs = np.array([np.squeeze(data) for _, data in weights.items()])  
        get_map(coefs, voxel_mask, task_name,
                map_type=map_type, save_path=save_path, sigma=sigma)
        
    report_function_dict['weights'] = get_map_2
    
    # TODO add pearson_r report
    
    return report_function_dict
    