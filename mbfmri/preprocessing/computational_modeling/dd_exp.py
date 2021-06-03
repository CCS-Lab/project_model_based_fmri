from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        r = param_dict["r"]
        beta  = param_dict["beta"]
    
        for amount_later,\
            amount_sooner,\
            delay_later,\
            delay_sooner in get_named_iterater(df_events,['amount_later',
                                                          'amount_sooner',
                                                          'delay_later',
                                                          'delay_sooner']):

            evLater = amount_later * exp(-1 * r * delay_later)
            evSooner  = amount_sooner * exp(-1 * r *  delay_sooner)
            suLater = evLater - evSooner
            suSooner = evSooner - evLater
            pLater= inv_logit(beta * (evLater - evSooner))
            pSooner = 1 - pLater
            
            self._add('EVlater', evLater)
            self._add('EVsooner', evSooner)
            self._add('SUlater', suLater)
            self._add('SUsooner', suSooner)
            self._add('Plater', pLater)
            self._add('Psooner', pSooner)

            
latent_process_onset = {}