from mbmvpa.utils.dataframe_utils import *


def function_PEchosen(df_events, param_dict):
    
    # get individual parameter values.
    eta_pos = float(param_dict['eta_pos'])
    eta_neg = float(param_dict['eta_neg'])
    
    modulations = [] # modulation
    
    # computational model part
    
    ev = [0,0]
    
    for choice, outcome in get_named_iterater(df_events,['choice',
                                                         'outcome']):
        choice = int(choice)
        outcome = int(outcome)
        
        PE  =  outcome - ev[choice-1];
        PEnc = -outcome - ev[2 - choice];
        
        if PE >= 0:
            ev[choice-1] += eta_pos * PE;
            ev[2 - choice] += eta_pos * PEnc;
        else :
            ev[choice-1] += eta_neg * PE;
            ev[2 - choice] += eta_neg * PEnc;
    
        modulations.append(PE) # add to modulation
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_PEnotchosen(df_events, param_dict):
    
    # get individual parameter values.
    eta_pos = float(param_dict['eta_pos'])
    eta_neg = float(param_dict['eta_neg'])
    
    modulations = [] # modulation
    
    # computational model part
    
    ev = [0,0]
    
    for choice, outcome in get_named_iterater(df_events,['choice',
                                                         'outcome']):
        choice = int(choice)
        outcome = int(outcome)
        
        PE  =  outcome - ev[choice-1];
        PEnc = -outcome - ev[2 - choice];
        
        if PE >= 0:
            ev[choice-1] += eta_pos * PE;
            ev[2 - choice] += eta_pos * PEnc;
        else :
            ev[choice-1] += eta_neg * PE;
            ev[2 - choice] += eta_neg * PEnc;
    
        modulations.append(PEnc) # add to modulation
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_EVchosen(df_events, param_dict):
    
    # get individual parameter values.
    eta_pos = float(param_dict['eta_pos'])
    eta_neg = float(param_dict['eta_neg'])
    
    modulations = [] # modulation
    
    # computational model part
    
    ev = [0,0]
    
    for choice, outcome in get_named_iterater(df_events,['choice',
                                                         'outcome']):
        choice = int(choice)
        outcome = int(outcome)
        
        PE  =  outcome - ev[choice-1];
        PEnc = -outcome - ev[2 - choice];
        
        if PE >= 0:
            ev[choice-1] += eta_pos * PE;
            ev[2 - choice] += eta_pos * PEnc;
        else :
            ev[choice-1] += eta_neg * PE;
            ev[2 - choice] += eta_neg * PEnc;
    
        modulations.append(ev[choice-1]) # add to modulation
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]

def function_EVnotchosen(df_events, param_dict):
    
    # get individual parameter values.
    eta_pos = float(param_dict['eta_pos'])
    eta_neg = float(param_dict['eta_neg'])
    
    modulations = [] # modulation
    
    # computational model part
    
    ev = [0,0]
    
    for choice, outcome in get_named_iterater(df_events,['choice',
                                                         'outcome']):
        choice = int(choice)
        outcome = int(outcome)
        
        PE  =  outcome - ev[choice-1];
        PEnc = -outcome - ev[2 - choice];
        
        if PE >= 0:
            ev[choice-1] += eta_pos * PE;
            ev[2 - choice] += eta_pos * PEnc;
        else :
            ev[choice-1] += eta_neg * PE;
            ev[2 - choice] += eta_neg * PEnc;
    
        modulations.append(ev[2-choice]) # add to modulation
        
    df_events["modulation"] = modulations
    
    return df_events[['onset','duration','modulation']]


latent_process_functions = {'PEchosen': function_PEchosen,
                            'PEnotchosen': function_PEnotchosen,
                           'EVchosen' : function_EVchosen,
                           'EVnotchosen' : function_EVnotchosen}

latent_process_onset = {'PEchosen': TIME_FEEDBACK,
                        'PEnotchosen': TIME_FEEDBACK}