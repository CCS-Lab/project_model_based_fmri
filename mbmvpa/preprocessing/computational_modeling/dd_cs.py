from mbmvpa.utils.dataframe_utils import *


def function_subjectiveutility(df_events, param_dict):
    
    # get individual parameter values.
    r = param_dict["r"]
    s = param_dict["s"]
    
    modulations = []
    
    for amount_later,\
        amount_sooner,\
        delay_later,\
        delay_sooner in get_named_iterater(df_events,['amount_later',
                                                      'amount_sooner',
                                                      'delay_later',
                                                      'delay_sooner']):
        
        # calculation here
        ev_later = amount_later * exp(-1 * pow(r * delay_later, s))
        ev_sooner  = amount_sooner * exp(-1 * pow(r * delay_sooner, s))
        modulation = ev_later - ev_sooner
        
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_evSooner(df_events, param_dict):
    
    # get individual parameter values.
    r = param_dict["r"]
    s = param_dict["s"]
    
    modulations = []
    
    for amount_sooner,\
        delay_sooner in get_named_iterater(df_events,['amount_sooner',
                                                      'delay_sooner']):
        
        # calculation here
        ev_sooner  = amount_sooner * exp(-1 * pow(r * delay_sooner, s))
        modulation = ev_sooner
        
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_evLater(df_events, param_dict):
    
    # get individual parameter values.
    r = param_dict["r"]
    s = param_dict["s"]
    
    modulations = []
    
    for amount_later,\
        delay_later in get_named_iterater(df_events,['amount_later',
                                                      'delay_later']):
        
        # calculation here
        ev_later = amount_later * exp(-1 * pow(r * delay_later, s))
        modulation = ev_later
        
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_pLater(df_events, param_dict):
    
    # get individual parameter values.
    r = param_dict["r"]
    s = param_dict["s"]
    beta  = param_dict["beta"]
    
    modulations = []
    
    for amount_later,\
        amount_sooner,\
        delay_later,\
        delay_sooner in get_named_iterater(df_events,['amount_later',
                                                      'amount_sooner',
                                                      'delay_later',
                                                      'delay_sooner']):
        
        # calculation here
        ev_later = amount_later * exp(-1 * pow(r * delay_later, s))
        ev_sooner  = amount_sooner * exp(-1 * pow(r * delay_sooner, s))
        modulation = inv_logit(beta * (ev_later - ev_sooner))
        
        modulations.append(modulation)
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]



latent_process_functions = {'subjectiveutility':function_subjectiveutility,
                           'evSooner' : function_evSooner,
                           'evLater' : function_evLater,
                           'pLater' : function_pLater}

latent_process_onset = {}