from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        alpha = param_dict['alpha']
        beta = param_dict['beta']
        gamma = param_dict['gamma']
        
        
        def _value(alpha, beta, p, a, v):
            return (p - beta * a / 2) * pow(v, alpha)
        

        for prob,ambig,reward_var,reward_fix,choice in get_named_iterater(df_events,['prob',
                                                                                    'ambig',
                                                                                    'reward_var',
                                                                                    'reward_fix',
                                                                                    'choice']):
            

            EVfix = _value(alpha, beta, 0.5, 0, reward_fix)
            EVvar = _value(alpha, beta, prob, ambig, reward_var)
            SUfix = EVfix - EVvar
            SUvar = EVvar - EVfix
            Pvar = inv_logit(gamma * SUvar)
            self._add('EVfix', EVfix)
            self._add('EVvar', EVvar)
            self._add('SUfix', SUfix)
            self._add('SUvar', SUvar)
            self._add('Pvar', Pvar)
            self._add('Pfix', 1-Pvar)
        
latent_process_onset = {}