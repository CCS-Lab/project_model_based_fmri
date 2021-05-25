from mbmvpa.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        A = param_dict['A']
        tau = param_dict['tau']
        
        ev = np.zeros(50)
        
        for choice,\
            outcome in get_named_iterater(df_events,['choice',
                                                    'outcome']):
            
            self._add('EVchosen',ev[choice-1])
            PE = outcome - ev[choice-1]
            self._add('PEchosen',PE)
            # value updating (learning)
            ev[choice-1] += A * PE
            
latent_process_onset = {'PEchosen': TIME_FEEDBACK}