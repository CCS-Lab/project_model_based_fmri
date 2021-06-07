from mbfmri.utils.computational_modeling_utils import *

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        Arew = param_dict['Arew']
        Apun = param_dict['Apun']
        R = param_dict['R']
        P = param_dict['P']
        xi = param_dict['xi']
        d = param_dict['d']
        
        Qr = np.zeros(50)
        Qp = np.zeros(50)
        Qsum = np.zeros(50)
        
        for choice,\
            gain,\
            loss in get_named_iterater(df_events,['choice',
                                                    'gain',
                                                    'loss']):
            
            loss = - abs(loss)
            Prob = softmax(Qsum) * (1-xi) + xi/4
            self._add('Pchosen',Prob[choice-1])
            
            # Prediction error signals
            PEr = R*gain - Qr[choice-1]
            PEp = P*loss - Qp[choice-1]
            self._add('PEreward',PEr)
            self._add('PEpunishment',PEp)
            self._add('PEchosen',PEr+PEp)
            #self._add('absPEchosen',abs(PEr+PEp))
            
            # store chosen deck Q values (rew and pun)
            Qr_chosen = Qr[choice-1]
            Qp_chosen = Qp[choice-1]
            
            self._add('QRchosen',Qr_chosen)
            self._add('QPchosen',Qp_chosen)
            self._add('Qchosen',Qr_chosen+Qp_chosen)
            # First, update Qr & Qp for all decks w/ fictive updating
            
            Qr = (1-d) * Qr
            Qp = (1-d) * Qp;
            
            # Replace Q values of chosen deck with correct values using stored values
            Qr[choice-1] = Qr_chosen + Arew * PEr
            Qp[choice-1] = Qp_chosen + Apun * PEp

            # Q(sum)
            Qsum = Qr + Qp
            
latent_process_onset = {'PEreward': TIME_FEEDBACK,
                       'PEpunishment': TIME_FEEDBACK,
                       'PEchosen':TIME_FEEDBACK,
                       #'absPEchosen':TIME_FEEDBACK
                       }