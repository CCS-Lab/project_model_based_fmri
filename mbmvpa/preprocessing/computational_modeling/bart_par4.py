from mbmvpa.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        phi =  param_dict["phi"]
        eta =  param_dict["eta"]
        gam = param_dict["gam"]
        tau = param_dict["tau"] 
        
        n_succ = 0
        n_pump = 0
        
        u_gain = 1
        
        for pumps, explosion in get_named_iterater(df_events,['pumps',
                                                              'explosion']):
            
            p_burst = 1 - ((phi + eta * n_succ) / (1 + eta * n_pump))
            omega = - gam / log1m(p_burst)
            
            n_succ += pumps - explosion
            n_pump += pumps
            
            self._add('Pburst', p_burst)
            self._add('Omega', omega)
            self._add('Expected Utility', (1-p_burst)^omega * (omega * u_gain)^gam)

            
latent_process_onset = {}
