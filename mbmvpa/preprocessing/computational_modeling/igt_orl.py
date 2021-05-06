from mbmvpa.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        Arew = param_dict['Arew']
        Apun = param_dict['Apun']
        K = param_dict['K']
        betaF = param_dict['betaF']
        betaP = param_dict['betaP'] 

        ef    = np.zeros(4) # [0,0,0,0]
        ev    = np.zeros(4) # [0,0,0,0]
        pers  = np.zeros(4) # [0,0,0,0]
        util  = np.zeros(4) # [0,0,0,0]
        K_tr = pow(3, K) - 1;

        for gain,\
            loss,\
            choice,\
            payscale in get_named_iterater(df_events,['gain',
                                                      'loss',
                                                      'choice',
                                                      'payscale'],{'payscale':100}):
            
            outcome = (gain - abs(loss))/payscale
            self._add('SUchosen', util[choice-1])
            PEval  = outcome - ev[choice-1];
            self._add('PEval', PEval)
            PEfreq = np.sign(outcome) - ef[choice-1]
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