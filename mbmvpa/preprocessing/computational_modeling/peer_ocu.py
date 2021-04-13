from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        rho = param_dict["rho"]
        tau = param_dict["tau"]
        ocu = param_dict["ocu"]

        for condition,\
            p_gamble,\
            safe_Hpayoff,\
            safe_Lpayoff,\
            risky_Hpayoff,\
            risky_Lpayoff,\
            choice in get_named_iterater(df_events,['condition',
                                                    'p_gamble',
                                                    'safe_Hpayoff',
                                                    'safe_Lpayoff',
                                                    'risky_Hpayoff',
                                                    'risky_Lpayoff',
                                                    'choice']):
    
                                                

            U_safe  = p_gamble * pow(safe_Hpayoff, rho) + \
                (1-p_gamble) * pow(safe_Lpayoff, rho)
            U_risky = p_gamble * pow(risky_Hpayoff, rho) + \
                (1-p_gamble) * pow(risky_Lpayoff, rho)
            if condition == 1: # safe-safe
                U_safe += ocu
            elif condition == 3: # risky-risky
                U_risky += ocu
                
            pRisky = inv_logit(tau * (U_risky - U_safe))

            self._add('EVsafe',U_safe)
            self._add('EVrisky',U_safe)
            self._add('SUrisky',U_risky - U_safe)
            self._add('Prisky',pRisky)
            self._add('Psafe', 1-Risky)
            
latent_process_onset = {}