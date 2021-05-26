from mbmvpa.utils.computational_modeling_utils import *

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
            
            self._add('Util_solo_safe',U_safe)
            self._add('Util_solo_risky',U_risky)
            
            if condition == 1: # safe-safe
                U_safe += ocu
            elif condition == 3: # risky-risky
                U_risky += ocu
                        
            pRisky = inv_logit(tau * (U_risky - U_safe))
            
            ##########################needs additional consideration for fMRI#######################
            if choice == 0: # modified utility: U_OCU chosen gamble − U_OCU unchosen gamble, for fMRI extraction 
                self._add('Util_unchosen_ocu',U_safe) # vmPFC
                self._add('Util_unchosen_ocu',U_risky) 
            elif choice == 1:
                self._add('Util_chosen_ocu',U_safe) # vmPFC  
                self._add('Util_chosen_ocu',U_risky)
            # U_ocu_safe and U_ocu_risky should be merged into U_ocu
                
            if condition == 1: # safe-safe
                self._add('distance',pRisky) # dACC, insula via interactive effect
            elif condition == 3: # risky-risky
                self._add('distance',1-pRisky) # dACC, insula via interactive effect
            # two distance_ss and distance_rr are essentially the same parameter
            #########################################################################################
            
            #self._add('Util_ocu_safe',U_safe)
            #self._add('Util_ocu_risky',U_risky)
            self._add('Prisky',pRisky)
            self._add('Psafe', 1-pRisky)
            
latent_process_onset = {}
