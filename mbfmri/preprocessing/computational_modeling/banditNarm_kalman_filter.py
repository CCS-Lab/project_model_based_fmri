from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        theta = param_dict['theta']
        beta = param_dict['beta']
        lambda_ = param_dict['lambda']
        mu0 = param_dict['mu0']
        sigma0 = param_dict['sigma0']
        sigmaD = param_dict['sigmaD']
        
        mu_ev = np.ones(50)*mu0
        sd_ev_sq = np.ones(50)*(sigma0**2)

        for choice,\
            outcome in get_named_iterater(df_events,['choice',
                                                    'outcome']):
            
            
            # learning rate
            k = sd_ev_sq[choice-1] / ( sd_ev_sq[choice-1] + sigma0**2 )
            # prediction error
            pe = outcome - mu_ev[choice-1]
            self._add('PEchosen',pe)
            # value updating (learning)
            self._add('EVchosen',mu_ev[choice-1])
            mu_ev[choice-1] += k * pe
            sd_ev_sq[choice-1] *= (1-k)
            # diffusion process
            mu_ev    *= lambda_;
            mu_ev    += (1 - lambda_) * theta
            sd_ev_sq *= lambda_**2
            sd_ev_sq += sigmaD**2
            
        
            
latent_process_onset = {'PEchosen': TIME_FEEDBACK}