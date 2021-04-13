from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        A = param_dict['A']
        alpha = param_dict['alpha']
        cons = param_dict['cons']
        lambda_ = param_dict['lambda']

        
        ev    = [0,0,0,0]
        theta = pow(3, cons) -1
        
        for outcome,\
            choice in get_named_iterater(df_events,['outcome',
                                                    'choice']):
            
            
            self._add('SUchosen', util[choice-1])
            self._add('EVchosen',ev[choice-1])
            
            if outcome >= 0: # x(t) >= 0
                curUtil = pow(outcome,alpha)
            else:
                curUtil = -1 * lambda_ * pow(-1 * outcome, alpha)
                
            self._add('Delta', curUtil)
            
            # decay-RI
            ev *= A
            ev[choice-1] += curUtil;
        
latent_process_onset = {'Delta', TIME_FEEDBACK}