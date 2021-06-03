from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        phi =  param_dict["phi"]
        eta =  param_dict["eta"]
        rho = param_dict["rho"]
        tau = param_dict["tau"] 
        lambda_ = param_dict["lambda"]
        
        u_gain = 1
        u_stop = 0
        p_burst = phi
        
        for pumps, explosion in get_named_iterater(df_events,['pumps',
                                                              'explosion']):
            
            for l in range(1, pumps-explosion+2):
                u_loss = l - 1
                u_pump = (1 - p_burst) * u_gain - lambda_ * p_burst * u_loss + \
                    rho * p_burst * (1 - p_burst) * (u_gain + lambda_ * u_loss)**2
                
            
            n_succ += pumps - explosion
            n_pump += pumps

            p_burst = phi + (1 - exp(-n_pump * eta)) * ((0.0 + n_pump - n_succ) / n_pump - phi)
            
            self._add('Pburst', p_burst)
            self._add('Upump', u_pump)

latent_process_onset = {}
