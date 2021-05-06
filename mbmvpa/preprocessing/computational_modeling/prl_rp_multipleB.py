from mbmvpa.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        Arew= param_dict['Arew']
        Apun = param_dict['Apun']
        beta = param_dict['beta']

        ev = [0,0]
        
        for choice, outcome in get_named_iterater(df_events,['choice',
                                                             'outcome']):
        
        
            self._add('EVchosen',ev[choice-1])
            self._add('EVnotchosen',ev[2 - choice])
            
            PE  =  outcome - ev[choice-1]
            
            self._add('PEchosen',PE)
            
            if outcome >= 0:
                ev[choice-1] += Arew * PE
            else :
                ev[choice-1] += Apun * PE
                
    
latent_process_onset = {'PEchosen': TIME_FEEDBACK,
                        'PEnotchosen': TIME_FEEDBACK}