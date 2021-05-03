#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Cheoljun cho
## contact: cjfwndnsl@gmail.com
## last modification: 2021.05.03

"""
This code is an interface for implementing Python classes for 
computational models.
Most of models implemented in hBayesDM are implemented. 
Refer to the codes in mbmvpa/preprocessing/computational_modeling/.
"""

class Base():
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
        