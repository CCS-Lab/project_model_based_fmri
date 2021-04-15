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
        
        for gain,\
            loss,\
            choice,\
            payscale in get_named_iterater(df_events,['gain',
                                                      'loss',
                                                      'choice',
                                                      'payscale'],{'payscale':100}):
            
            outcome = (gain - abs(loss))/payscale
            self._add('EVchosen',ev[choice-1])
            
            if outcome >= 0: # x(t) >= 0
                curUtil = pow(outcome,alpha)
            else:
                curUtil = -1 * lambda_ * pow(-1 * outcome, alpha)
                
            self._add('curUtil', curUtil)
            
            # delta
            ev[choice-1] += A * (curUtil - ev[choice-1])
        
latent_process_onset = {'curUtil', TIME_FEEDBACK}