from mbfmri.utils.computational_modeling_utils import *

# Not implemented due to complex preprocessing in hBayesDM

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        alpha = param_dict['alpha']
        rho = param_dict['rho']
        gamma = param_dict['gamma']
        c = param_dict['c']
        beta = param_dict['beta']
        
        
        for gamble_type,\
            percentage_staked,\
            trial_initial_points,\
            assessment_stage,\
            red_chosen,\
            n_red_boxes in get_named_iterater(df_events,['gamble_type',
                                                        'percentage_staked',
                                                        'trial_initial_points',
                                                        'assessment_stage',
                                                        'red_chosen',
                                                        'n_red_boxes']):
            
            x = 1
            
latent_process_onset = {'PEchosen': TIME_FEEDBACK}