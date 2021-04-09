from utils.dataframe_utils import get_named_iterater, inv_logit

def function_subjectiveutility(df_events, param_dict):
    
    # get individual parameter values.
    rho = param_dict["rho"] 
    lambda_ = param_dict["lambda"]
    
    modulations = []
    
    for gain,loss,cert in get_named_iterater(df_events,['gain','loss','cert']):
        
        # calculation here
        evSafe   = cert**rho
        evGamble = 0.5 * (gain**rho - lambda_ *(loss**rho))
        modulation = evGamble - evSafe
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_evGamble(df_events, param_dict):
    
    # get individual parameter values.
    rho = param_dict["rho"] 
    lambda_ = param_dict["lambda"]
    
    modulations = []
    
    for gain,loss,cert in get_named_iterater(df_events,['gain','loss','cert']):
        
        # calculation here
        evGamble = 0.5 * (gain**rho - lambda_ *(loss**rho))
        modulation = evGamble
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_evSafe(df_events, param_dict):
    
    # get individual parameter values.
    rho = param_dict["rho"] 
    lambda_ = param_dict["lambda"]
    
    modulations = []
    
    for gain,loss,cert in get_named_iterater(df_events,['gain','loss','cert']):
        
        # calculation here
        evSafe = cert**rho
        modulation = evSafe
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_pGamble(df_events, param_dict):
    
    # get individual parameter values.
    rho = param_dict["rho"] 
    lambda_ = param_dict["lambda"]
    tau = param_dict["tau"]
    
    modulations = []
    
    for gain,loss,cert in get_named_iterater(df_events,['gain','loss','cert']):
        
        # calculation here
        evSafe   = cert**rho
        evGamble = 0.5 * (gain**rho - lambda_ *(loss**rho))
        modulation = inv_logit(tau*(evGamble - evSafe))
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]




latent_process_functions = {'subjectiveutility':function_subjectiveutility,
                           'evGamble': function_evGamble,
                           'evGafe': function_evSafe,
                           'pGamble': function_pGamble}