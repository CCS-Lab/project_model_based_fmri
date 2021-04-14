from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        A = param_dict['A']
        tau = param_dict['tau']
        
        ev = [0,0]
        
        for choice,\
            outcome in get_named_iterater(df_events,['choice',
                                                    'outcome']):
            
            

            self._add('EVchosen',ev[choice-1])
            PE = outcome - ev[choice-1]
            self._add('PEchosen',PE[choice-1])
            # value updating (learning)
            ev[choice-1] += A * PE
            
latent_process_onset = {'PEchosen': TIME_FEEDBACK}