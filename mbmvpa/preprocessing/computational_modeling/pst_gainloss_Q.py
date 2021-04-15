from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
    
        # get individual parameter values.
        alpha_pos = param_dict["alpha_pos"]
        alpha_neg = param_dict["alpha_neg"]
        betas = param_dict["beta"]
        
        ev = [0] *6
    
        for type_, choice, reward in get_named_iterater(df_events,['type',
                                                                    'choice',
                                                                    'reward']):
            
            option1 = type_ // 10
            option2 = type_ % 10
            if choice > 0:
                co = option1
            else:
                co = option2
            self._add('EV',ev[co-1])
        
            # Luce choice rule
            delta = ev[option1-1] - ev[option2-1]
            pe = reward - ev[co-1]
            alpha = alpha_pos if pe >= 0 else alpha_neg
            ev[co-1] += alpha * pe
            self._add('PE',pe)
            self._add('Delta',delta)
        

latent_process_onset = {'PE': TIME_FEEDBACK,
                        'Delta': TIME_FEEDBACK}