from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        eta_pos = param_dict['eta_pos']
        eta_neg = param_dict['eta_neg']

        ev = [0,0]

        for choice, outcome in get_named_iterater(df_events,['choice',
                                                             'outcome']):
            
            self._add('EVchosen',ev[choice-1])
            self._add('EVnotchosen',ev[2 - choice])
            
            PE  =  outcome - ev[choice-1]
            PEnc = -outcome - ev[2 - choice]
            
            self._add('PEchosen',PE)
            self._add('PEnotchosen',PEnc)
            
            if PE >= 0:
                ev[choice-1] += eta_pos * PE
                ev[2 - choice] += eta_pos * PEnc
            else :
                ev[choice-1] += eta_neg * PE
                ev[2 - choice] += eta_neg * PEnc


latent_process_onset = {'PEchosen': TIME_FEEDBACK,
                        'PEnotchosen': TIME_FEEDBACK}