from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        w0 = param_dict['w0']
        w1 = param_dict['w1']
        w2 = param_dict['w2']
        w3 = param_dict['w3']
        gam = param_dict['gam']
        sig = param_dict['sig']
        
        cert_sum = 0
        ev_sum = 0
        rpe_sum = 0

        for gain,\
            loss,\
            cert,\
            type_,\
            gamble,\
            outcome in get_named_iterater(df_events,['gain',
                                                      'loss',
                                                      'cert',
                                                      'type',
                                                      'gamble',
                                                      'outcome',
                                                      ]):
            if gamble == 0:
                ev = type_ * cert
                cert_sum += ev
                rpe = 0
            else:
                ev = 0.5 * (gain - loss)
                ev_sum += ev
                rpe = outcome - 0.5 * (gain - loss)
                rpe_sum += rpe
            
            cert_sum *= gam
            ev_sum   *= gam
            rpe_sum  *= gam
            
            self._add('PE',rpe)
            self._add('EV',ev)
            self._add('PEsum',rpe_sum)
            self._add('EVsum',ev_sum)
            sefl._add('Certsum',cert_sum)

latent_process_onset = {'PE': TIME_FEEDBACK,
                        'PEsum}': TIME_FEEDBACK}