from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        a = param_dict['a']
        beta = param_dict['beta']
        pi = param_dict['pi']
        w = param_dict['w']
        
        v_mb = [0] *2
        v_mf  = [0] *6
        v_hybrid = [0] *2
        
        t = 0
        
        for level1_choice,\
            level2_choice,\
            reward in get_named_iterater(df_events,['level1_choice',
                                                    'level2_choice',
                                                    'reward',
                                                    'trans_prob'],{'trans_prob':0.7}):
            
            t += 1
            
            v_mb[0] = trans_prob * fmax(v_mf[2], v_mf[3]) + (1 - trans_prob) * fmax(v_mf[4], v_mf[5]) # for level1, stim 1
            v_mb[1] = (1 - trans_prob) * fmax(v_mf[2], v_mf[3]) + trans_prob * fmax(v_mf[4], v_mf[5]) # for level1, stim 2

            # compute v_hybrid
            v_hybrid[0] = w * v_mb[0] + (1-w) * v_mf[0]   # hybrid stim 1= weighted sum
            v_hybrid[1] = w * v_mb[1] + (1-w) * v_mf[1]   # hybrid stim 2= weighted sum

            # Prob of choosing stimulus 2 in ** Level 1 ** --> to be used on the next trial
            # level1_choice=1 --> -1, level1_choice=2 --> 1
            
            level1_choice_01 = level1_choice - 1  # convert 1,2 --> 0,1
            if t == 1:
                level1_prob_choice2 = inv_logit( beta*(v_hybrid[1]-v_hybrid[0]))
            else:
                level1_prob_choice2 = inv_logit( beta*(v_hybrid[1]-v_hybrid[0]) + pi*(2*prev_level1_choice -3) )
    
            # Observe Level2 and update Level1 of the chosen option
            v_mf[level1_choice-1] += a*(v_mf[1+ level2_choice] - v_mf[level1_choice-1])

            # Prob of choosing stim 2 (2 from [1,2] OR 4 from [3,4]) in ** Level (step) 2 **
            level2_choice_01 = 1 - (level2_choice[i,t] % 2) # 1,3 --> 0; 2,4 --> 1
            
            if level2_choice > 2:  # level2_choice = 3 or 4
                level2_prob_choice2 = inv_logit( beta*( v_mf[5] - v_mf[4] ) )
            else : # level2_choice = 1 or 2
                level2_prob_choice2 = inv_logit( beta*( v_mf[3] - v_mf[2] ) )
           
            # After observing the reward at Level 2...
            # Update Level 2 v_mf of the chosen option. Level 2--> choose one of level 2 options and observe reward
            v_mf[1+ level2_choice] += a*(reward - v_mf[1+ level2_choice ])

            # Update Level 1 v_mf
            v_mf[level1_choice] += a * (reward - v_mf[1+level2_choice])
            
            prev_lavel1_choice = level1_choice
            
latent_process_onset = {'PE': TIME_FEEDBACK,
                        'PEsum}': TIME_FEEDBACK}