from mbmvpa.utils.dataframe_utils import *
from .base_model import Base

class ComputationalModel(Base):
    def _set_latent_process(self, df_events, param_dict):
        
        tau = param_dict['tau']
        rho = param_dict['rho']
        lambda_ = param_dict['lambda']
        beta = param_dict['beta']
        
        U_opt = [0,0]
        
        for opt1hprob,\
            opt2hprob,\
            opt1hval,\
            opt1lval,\
            opt2hval,\
            opt2lval,\
            choice in get_named_iterater(df_events,['opt1hprob',
                                                   'opt2hprob',
                                                   'opt1hval',
                                                   'opt1lval',
                                                   'opt2hval',
                                                   'opt2lval',
                                                    'choice']):
            
            w_prob = [exp(-(-log(opt1hprob))**tau),
                      exp(-(-log(1-opt1hprob))**tau),
                      exp(-(-log(opt2hprob))**tau),
                      exp(-(-log(1-opt2hprob))**tau)]
            
            if opt1hval > 0:
                if opt1lval >= 0:
                    U_opt[0] = w_prob[0]*(opt1hval**rho) + w_prob[1]*(opt1lval**rho)
                else:
                    U_opt[0] =  w_prob[0]*(opt1hval**rho) - w_prob[1]* (abs(opt1lval)**rho)*lambda_
            else:
                U_opt[0] = -w_prob[0]*(abs(opt1hval)**rho)*lambda_- w_prob[1]*(abs(opt1lval)^rho)*lambda_
            
            if opt2hval > 0:
                if opt2lval >= 0:
                    U_opt[1]  = w_prob[2]*(opt2hval**rho) + w_prob[3]*(opt2lval**rho)
                else:
                    U_opt[1] = w_prob[2]*(opt2hval**rho) - w_prob[3]*(abs(opt2lval)**rho)*lambda_
            else:
                U_opt[1] = -w_prob[2]*(abs(opt2hval)**rho)*lambda_ - w_prob[3]*(abs(opt2lval)**rho)*lambda_
            
            P_choice = inv_logit(beta*U_opt[choice-1])
            
            self._add('EVchosen', U_opt[choice-1])
            self._add('EVnotchosen', U_opt[2-choice])
            self._add('SUchosen',  U_opt[choice-1] - U_opt[2-choice])
            self._add('SUnotchosen', - U_opt[choice-1] + U_opt[2-choice])
            self._add('Pchoice',P_choice)
        
latent_process_onset = {}