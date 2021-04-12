from mbmvpa.utils.dataframe_utils import get_named_iterater, inv_logit, TIME_FEEDBACK

def function_processname1(df_events, param_dict):
    
    # get individual parameter values.
    alpha = param_dict["alpha"] 
    beta = param_dict["beta"]
    
    modulations = []
    
    for a,b,c in get_named_iterater(df_events,['a','b','c']):
        
        # calculation here
        
        modulation = a+b+c #
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]


def function_processname2(df_events, param_dict):
    
    # get individual parameter values.
    alpha = param_dict["alpha"] 
    beta = param_dict["beta"]
    
    modulations = []
    
    for a,b,c in get_named_iterater(df_events,['a','b','c']):
        
        # calculation here
        
        modulation = a+b+c #
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]



latent_process_functions = {'processname1':function_processname1,
                            'processname2':function_processname2,}


latent_process_onset = {'processname1': TIME_FEEDBACK}