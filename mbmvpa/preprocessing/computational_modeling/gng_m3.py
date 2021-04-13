from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        xi = param_dict["xi"]
        ep = param_dict["ep"]
        rho = param_dict["rho"]
        b = param_dict["b"]
        pi = param_dict["pi"]
        
        wv_g  = [0, 0, 0, 0]
        wv_ng = [0, 0, 0, 0]
        qv_g  = [0, 0, 0, 0]
        qv_ng = [0, 0, 0, 0]
        pGo = [0, 0, 0, 0]
        sv = [0, 0, 0, 0]
    
        for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                    'keyPressed',
                                                                    'outcome']):

            self._add('SV',sv[cue-1])
            self._add('QVgo', qv_g[cue-1])
            wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
            self._add('WVgo', wv_g[cue-1])
            
            self._add('QVnogo', qv_ng[cue-1])
            wv_ng[cue-1] = qv_ng[cue-1]  
            self._add('WVnogo', wv_ng[cue-1])
            
            self._add('SUgo', wv_g[cue-1] - wv_ng[cue-1])
            self._add('SUnogo', - wv_g[cue-1] + wv_ng[cue-1])
            
            pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
            pGo[cue-1] *= (1 - xi)
            pGo[cue-1] += (xi/2)
            
            self._add('Pgo',Pgo)
            
            sv[cue-1] += (ep * (rho * outcome - sv[cue-1]))
            
            if keyPressed == 1:
                PEgo = rho * outcome - qv_g[cue-1]
                PEnogo = 0
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEgo = 0
                PEnogo = rho * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                                 
            self._add('PEgo', PEgo)
            self._add('PEnogo', PEnogo)
            
latent_process_onset = {'PEgo': TIME_FEEDBACK,
                       'PEnogo': TIME_FEEDBACK}