from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        phi = param_dict['phi']
        rho = param_dict['rho']
        beta = param_dict['beta']
        
        ev = np.zeros(2) # [0,0]
        ew = np.zeros(2) # [0,0]

        for choice, outcome in get_named_iterater(df_events,['choice',
                                                             'outcome']):
            
            # Store previous experience weight value
            ewt1 = ew[choice-1];
            
            self._add('EVchosen',ev[choice-1])
            self._add('EVnotchosen',ev[2 - choice])
            
            # Update experience weight for chosen stimulus
            ew[choice-1] *= rho
            ew[choice-1] += 1

            # Update expected value of chosen stimulus
            ev[choice-1] *= phi * ewt1
            ev[choice-1] += outcome
            ev[choice-1] /= ew[choice-1]


latent_process_onset = {}