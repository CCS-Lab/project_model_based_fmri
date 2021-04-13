from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        k = param_dict["k"]
        beta  = param_dict["beta"]
        
        for amount_later,\
            amount_sooner,\
            delay_later,\
            delay_sooner in get_named_iterater(df_events,['amount_later',
                                                          'amount_sooner',
                                                          'delay_later',
                                                          'delay_sooner']):

            evLater = amount_later / (1 + k * delay_later)
            evSooner  = amount_sooner / (1 + k * delay_sooner)
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