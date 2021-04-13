from mbmvpa.utils.dataframe_utils import * #Do not change

# function for process 1
def function_PEgo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                        
                                 
        modulation = PEgo 
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

# function for process 2
def function_PEnogo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                                 
        modulation = PEnogo 
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

def function_QVgo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = qv_g[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                                 
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change


def function_QVnogo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = qv_ng[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

def function_WVgo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = wv_g[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                                 
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change


def function_WVnogo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = wv_ng[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

def function_subjectiveutility(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = wv_g[cue-1] - wv_ng[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                                 
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

def function_pGo(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = pGo[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
                                 
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

def function_stimulusvalue(df_events, param_dict):
    
    # get individual parameter values.
    xi = param_dict["xi"]
    ep = param_dict["ep"]
    rhoRew = param_dict["rhoRew"]
    rhoPun = param_dict["rhoPun"]
    b = param_dict["b"]
    pi = param_dict["pi"]
    
    modulations = [] # do not change
    
    wv_g  = [0, 0, 0, 0]
    wv_ng = [0, 0, 0, 0]
    qv_g  = [0, 0, 0, 0]
    qv_ng = [0, 0, 0, 0]
    pGo = [0, 0, 0, 0]
    sv = [0, 0, 0, 0]
    
    for cue, keyPressed, outcome in get_named_iterater(df_events,['cue',
                                                                'keyPressed',
                                                                'outcome']):
        
        # calculation here
        # python code for "model" part in corresponding stan file
        
        wv_g[cue-1] = qv_g[cue-1] + b + pi * sv[cue-1]
        wv_ng[cue-1] = qv_ng[cue-1]  
        pGo[cue-1] = inv_logit(wv_g[cue-1] - wv_ng[cue-1])
        pGo[cue-1] *= (1 - xi)
        pGo[cue-1] += (xi/2)
        
        modulation = sv[cue-1]
        
        PEnogo = 0
        PEgo = 0
        
        if outcome >= 0 :
            sv[cue-1] += (ep * (rhoRew * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoRew * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoRew * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        else:
            sv[cue-1] += (ep * (rhoPun * outcome - sv[cue-1]))
            if keyPressed == 1:
                PEgo = rhoPun * outcome - qv_g[cue-1]
                qv_g[cue-1] += (ep * PEgo)
            else:
                PEnogo = rhoPun * outcome - qv_ng[cue-1]
                qv_ng[cue-1] += (ep * PEnogo)
        
        modulations.append(modulation) # do not change
        
    df_events["modulation"] = modulations # do not change
    
    return df_events[['onset','duration','modulation']]  # do not change

#  name - function dictionary. please match the name.
latent_process_functions = {'PEgo':function_PEgo,
                            'PEnogo':function_PEnogo,
                           'QVgo': function_QVgo,
                           'QVnogo': function_QVnogo,
                           'WVgo': function_WVgo,
                           'WVnogo': function_WVnogo,
                           'subjectiveutility': function_subjectiveutility,
                           'pGo': function_pGo,
                           'stimulusvalue': function_stimulusvalue}

# name - onset info. 
# if the process is meaningful at the time of feedback in the corresponding task.
# So, indicate it with the below "latent_process_onset" dictionary.
# e.g. "processname1" is the process effective at the feedback. 
# other processes not indicated (here processname2) would be regarded effective at "onset."
latent_process_onset = {'PEgo': TIME_FEEDBACK,
                       'PEnogo': TIME_FEEDBACK,
                       'stimulusvalue': TIME_FEEDBACK}