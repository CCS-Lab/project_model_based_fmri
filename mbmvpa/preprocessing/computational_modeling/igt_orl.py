from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        Arew = param_dict['Arew']
        Apun = param_dict['Apun']
        K = param_dict['K']
        betaF = param_dict['betaF']
        betaP = param_dict['betaP'] 

        ef    = [0,0,0,0]
        ev    = [0,0,0,0]
        pers  = [0,0,0,0]
        util  = [0,0,0,0]
        K_tr = pow(3, K) - 1;

        for outcome,\
            choice in get_named_iterater(df_events,['outcome',
                                                    'choice']):
            
            self._add('SUchosen', util[choice-1])
            PEval  = outcome - ev[choice-1];
            self._add('PEval', PEval)
            PEfreq = sign_out[i,t] - ef[ choice-1]
            self._add('PEfreq', PEfreq)
            PEfreq_fic = -sign_out(gain,loss)/3 - ef
            
            efChosen = ef[choice-1]
            evChosen = ev[choice-1]
            self._add('EFchosen',efChosen)
            self._add('EVchosen',evChosen)
            self._add('PEval', PEval)
            
            if outcome >= 0:
                # Update ev for all decks
                ef += Apun * PEfreq_fic
                # Update chosendeck with stored value
                ef[choice-1] = efChosen + Arew * PEfreq
                ev[choice-1] = evChosen + Arew * PEval
            else :
                # Update ev for all decks
                ef += Arew * PEfreq_fic
                # Update chosendeck with stored value
                ef[choice-1] = efChosen + Apun * PEfreq
                ev[choice-1] = evChosen + Apun * PEval
            
            # Perseverance updating
            pers[choice-1] = 1   # perseverance term
            pers /= (1 + K_tr)   # decay

            # Utility of expected value and perseverance
            util  = ev + ef * betaF + pers * betaP;
            
        
latent_process_onset = {'PEval', TIME_FEEDBACK,
                       'PEfreq', TIME_FEEDBACK}