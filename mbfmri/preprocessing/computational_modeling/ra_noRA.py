from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        lambda_ = param_dict["lambda"]
        tau = param_dict["tau"]

        for gain,loss,cert in get_named_iterater(df_events,['gain','loss','cert']):
            
            loss = abs(loss)
            # calculation here
            evSafe   = cert
            evGamble = 0.5 * (gain - lambda_ *loss)
            subjectiveutility = evGamble - evSafe
            pGamble = inv_logit(tau*(evGamble - evSafe))

            self._add('EVsafe',evSafe)
            self._add('EVgamble',evGamble)
            self._add('SUgamble',subjectiveutility)
            self._add('SUsave', -subjectiveutility)
            self._add('Pgamble', pGamble)
            self._add('Psafe', 1-pGamble)
            
latent_process_onset = {}