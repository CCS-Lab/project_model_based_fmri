from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        A = param_dict['A']
        alpha = param_dict['alpha']
        cons = param_dict['cons']
        lambda_ = param_dict['lambda']
        epP = param_dict['epP']
        K = param_dict['K']
        w = param_dict['w']
        
        
        ev    = [0,0,0,0]
        pers = [0,0,0,0]
        V = [0,0,0,0]
        theta = pow(3, cons) -1
        
        for outcome,\
            choice in get_named_iterater(df_events,['outcome',
                                                    'choice']):
            
            
                
                self._add('SUchosen', V[choice-1])
                self._add('EVchosen',ev[choice-1])
            
                # perseverance decay
                pers *= K # decay
                
                if outcome >= 0: # x(t) >= 0
                    curUtil = pow(outcome, alpha)
                    pers[choice-1] += epP  # perseverance term
                else: # x(t) < 0
                    curUtil = -1 * lambda_ * pow(-1 * outcome, alpha)
                    pers[choice-1] += epN  # perseverance term
              
                update = A * (curUtil - ev[choice-1])
                self._add('Delta',update)
                ev[choice-1] += A * (curUtil - ev[choice-1])
                # calculate V
                V = w[i] * ev + (1-w[i]) * pers;
            
        
latent_process_onset = {'Delta', TIME_FEEDBACK}