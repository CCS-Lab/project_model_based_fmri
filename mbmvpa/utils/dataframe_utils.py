import pandas as pd
import numpy as np

TIME_ONSET = "onset"
TIME_FEEDBACK = "time_feedback"
np = np

def get_named_iterater(df_events, name_list, default={}):
    iter_list = []
    for name in name_list:
        if name not in df_events.columns:
            iter_list.append([default[name]]*len(df_events))
        else:
            iter_list.append(df_events[name].to_numpy())
            
    return zip(*iter_list)

def inv_logit(p):
    return np.exp(p) / (1 + np.exp(p))

def exp(x):
    return np.exp(x)

def log1m(x):
    return np.log(1-x)

def sign_out(gain,loss):
    return np.sign(gain-np.abs(loss))

def log(x):
    return np.log(x)

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x