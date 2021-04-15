from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        alpha = param_dict['alpha']
        ep = param_dict['ep']
        tau = param_dict['tau']
        
        f   = 10.0
        
        for offer,\
            accept in get_named_iterater(df_events,['offer',
                                                    'accept']):
            
            self._add('Norm',f)
            # calculate prediction error
            PE = offer - f

            # Update utility
            util = offer - alpha * max(f - offer, 0.0)
            self._add('Util',util)
            # Update internal norm
            f += ep * PE
            self._add('PE',PE)
           
            
latent_process_onset = {'PE': TIME_FEEDBACK}