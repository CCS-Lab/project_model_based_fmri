from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        Arew = param_dict['Arew']
        Apew = param_dict['Apew']
        R = param_dict['R']
        P = param_dict['P']
        xi = param_dict['xi']
        
        Qr = np.zeros(4)
        Qp = np.zeros(4)
        Qsum = np.zeros(4)
        
        for choice,\
            gain,\
            loss in get_named_iterater(df_events,['choice',
                                                    'gain',
                                                    'loss']):
            
            loss = - abs(loss)
            Prob = softmax(Qsum) * (1-xi) + xi/4
            self._add('Pchosen',Prob[choice-1])
            
            # Prediction error signals
            PEr = R*gain - Qr[-1]
            PEp = P*loss - Qp[choice-1]
            self._add('PEreward',PEr)
            self._add('PEpunishment',PEp)
            
            PEr_fic = -Qr
            PEp_fic = -Qp

            # store chosen deck Q values (rew and pun)
            Qr_chosen = Qr[choice-1]
            Qp_chosen = Qp[choice-1]
            
            self._add('QRchosen',Qr_chosen)
            self._add('QPchosen',Qp_chosen)
            
            # First, update Qr & Qp for all decks w/ fictive updating
            Qr += Arew * PEr_fic
            Qp += Apun * PEp_fic
            # Replace Q values of chosen deck with correct values using stored values
            Qr[choice-1] = Qr_chosen + Arew * PEr
            Qp[choice-1] = Qp_chosen + Apun * PEp

            # Q(sum)
            Qsum = Qr + Qp
            
latent_process_onset = {'PEreward': TIME_FEEDBACK,
                       'PEpunishment': TIME_FEEDBACK}