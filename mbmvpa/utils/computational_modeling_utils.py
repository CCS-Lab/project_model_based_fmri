#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.05.03
                         

import pandas as pd
import numpy as np

TIME_ONSET = "onset"
TIME_FEEDBACK = "time_feedback"
np = np

class Base():
    """
    This class is an interface for implementing Python classes for 
    computational models.
    Most of models implemented in hBayesDM are implemented. 
    Refer to the codes in mbmvpa/preprocessing/computational_modeling/.
    """
    def __init__(self, process_name):
        self.process_name = process_name
        self.latent_process = {}
    
    def _set_latent_process(self, df_events, param_dict):
        # implement
        return
    
    def _add(self, key, value):
        if key not in self.latent_process.keys():
            self.latent_process[key] = []
        self.latent_process[key].append(value)
    
    def __call__(self, df_events, param_dict):
        self.latent_process = {}
        self._set_latent_process(df_events, param_dict)
        df_events["modulation"] = self.latent_process[self.process_name]
        return df_events[['onset','duration','modulation']]
                         
                         
# get iterater from DataFrame with designated column names.
def get_named_iterater(df_events, name_list, default={}):
    iter_list = []
    for name in name_list:
        if name not in df_events.columns:
            # if name is absent, use default value instead.
            iter_list.append([default[name]]*len(df_events))
        else:
            iter_list.append(df_events[name].to_numpy())
            
    return zip(*iter_list)

# arithmetic helper functions

def inv_logit(p):
    p = float(p)
    return np.exp(p) / (1 + np.exp(p))

def exp(x):
    x = float(x)
    return np.exp(x)

def log1m(x):
    x = float(x)
    return np.log(1-x)

def sign_out(gain,loss):
    gain = float(gain)
    loss = float(loss)
    return np.sign(gain-np.abs(loss))

def log(x):
    x = float(x)
    return np.log(x)

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x