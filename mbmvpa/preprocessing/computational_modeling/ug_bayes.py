from mbmvpa.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        alpha = param_dict['alpha']
        beta = param_dict['beta']
        tau = param_dict['tau']
        
        mu_old   = 10.0
        k_old    = 4.0
        sig2_old = 4.0
        nu_old   = 10.0
        
        for offer,\
            accept in get_named_iterater(df_events,['offer',
                                                    'accept']):
            
            k_new  = k_old + 1
            nu_new = nu_old +1
            mu_new = (k_old/k_new) * mu_old + (1/k_new) * offer
            sig2_new = (nu_old/nu_new) * sig2_old + (1/nu_new) * (k_old/k_new) * pow((offer - mu_old), 2)
            PE   = offer - mu_old
            self._add('PE',PE)
            util = offer - alpha * max(mu_new - offer, 0.0) - beta * max(offer - mu_new, 0.0)
            self._add('Util',util)
            mu_old   = mu_new
            sig2_old = sig2_new
            k_old    = k_new
            nu_old   = nu_new
            
latent_process_onset = {'PE': TIME_FEEDBACK}