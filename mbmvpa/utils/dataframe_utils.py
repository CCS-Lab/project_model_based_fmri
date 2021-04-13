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