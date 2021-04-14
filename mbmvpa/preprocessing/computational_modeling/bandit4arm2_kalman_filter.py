from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        theta = param_dict['theta']
        beta = param_dict['beta']
        lambda_ = param_dict['lambda']
        mu0 = param_dict['mu0']
        sigma0 = param_dict['sigma0']
        sigmaD = param_dict['sigmaD']
        
        mu_ev = np.ones(4)*mu0
        sd_ev_sq = np.ones(4)*(sigma0[i]**2)

        for choice,\
            outcome in get_named_iterater(df_events,['choice',
                                                    'outcome']):
            
            
            # learning rate
            k = sd_ev_sq[choice-1] / ( sd_ev_sq[choice-1] + sigmaO**2 )
            # prediction error
            pe = outcome - mu_ev[choice-1]
            self._set('PEchosen',pe)
            # value updating (learning)
            self._set('EVchosen',mu_ev[choice-1])
            mu_ev[choice-1] += k * pe
            sd_ev_sq[choice-1] *= (1-k)
            # diffusion process
            mu_ev    *= lambda_;
            mu_ev    += (1 - lambda_) * theta
            sd_ev_sq *= lambda_**2
            sd_ev_sq += sigmaD**2
            
        
            
latent_process_onset = {'PEchosen': TIME_FEEDBACK}