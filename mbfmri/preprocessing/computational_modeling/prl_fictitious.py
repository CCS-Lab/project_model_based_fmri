from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        eta = param_dict['eta']
        alpha = param_dict['alpha']
        beta = param_dict['beta']

        ev = [0,0]
        prob = [0,0]

        for choice, outcome in get_named_iterater(df_events,['choice',
                                                             'outcome']):
            # Compute action probabilities
            prob[0] = 1 / (1 + exp(beta * (alpha - (ev[0] - ev[1]))))
            prob_1_ = prob[0]
            prob[1] = 1 - prob_1_;
            
            self._add('EVchosen',ev[choice-1])
            self._add('EVnotchosen',ev[2 - choice])
            
            
            # Prediction error
            PE =  outcome - ev[choice-1]
            PEnc = -outcome - ev[2-choice]
            
            self._add('PEchosen',PE)
            self._add('PEnotchosen',PEnc)
            
            # Value updating (learning)
            ev[choice-1]   += eta * PE
            ev[2-choice] += eta * PEnc
            

latent_process_onset = {'PEchosen': TIME_FEEDBACK,
                        'PEnotchosen': TIME_FEEDBACK}