#Do not change
from mbmvpa.utils.dataframe_utils import get_named_iterater, inv_logit, TIME_FEEDBACK

# function for process 1
def function_processname1(df_events, param_dict):
    
    # get individual parameter values.
    alpha = param_dict["alpha"] 
    beta = param_dict["beta"]
    
    modulations = [] # do not change
    
    for a,b,c in get_named_iterater(df_events,['a','b','c']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        modulation = a+b+c #
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

# function for process 2
def function_processname2(df_events, param_dict):
    
    # get individual parameter values.
    alpha = param_dict["alpha"] 
    beta = param_dict["beta"]
    
    modulations = [] # do not change
    
    for a,b,c in get_named_iterater(df_events,['a','b','c']):
        
        # calculation here
        # python code for "model" part in corresponding stan file.
        modulation = a+b+c #
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']] # do not change



#  name - function dictionary. please match the name.
latent_process_functions = {'processname1':function_processname1,
                            'processname2':function_processname2,}

# name - onset info. 
# if the process is meaningful at the time of feedback in the corresponding task.
# So, indicate it with the below "latent_process_onset" dictionary.
# e.g. "processname1" is the process effective at the feedback. 
# other processes not indicated (here processname2) would be regarded effective at "onset."
latent_process_onset = {'processname1': TIME_FEEDBACK}