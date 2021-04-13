import pandas as pd
import numpy as np

TIME_ONSET = "onset"
TIME_FEEDBACK = "time_feedback"

def get_named_iterater(df_events, name_list):
  
    return zip(*[df_events[name].to_numpy() for name in name_list])

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