#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.03

import mbfmri.utils.config
import importlib

class MBFMRI():
    
    r"""
        
    """
    def __init__(self,
                 config=None,
                 **kwargs):
        
        # load & set configuration
        self.config = self._load_default_config()
        self._override_config(config)
        self._add_kwargs_to_config(kwargs)
        
    def _load_default_config(self):
        importlib.reload(mbfmri.utils.config)
        return mbfmri.utils.config.DEFAULT_ANALYSIS_CONFIGS
    
    def _override_config(self,config):
        """
        config should be a dictionary or yaml file.
        configuration handled in this class is a tree-like dictionary.
        override the default configuration with input config.
        """
        if config is None:
            return
        if isinstance(config, str):
            config = yaml.load(open(config))
            
        def override(a, b):
            # recursive function
            for k,d in b.items():
                
                if isinstance(d,dict):
                    if k in a.keys():
                        override(a[k],d)
                    else:
                        a[k] = d
                else:
                    a[k] = d
        
        override(self.config,config)
        
    def _add_kwargs_to_config(self,kwargs):
        """
        override configuration dictionary with keywarded arguments.
        find the leaf node in configuration tree which match the keyward.
        then override the value.
        """
        added_keywords = []
        def recursive_add(kwargs,config):
            # recursive function
            if not isinstance(config,dict):
                return 
            else:
                for k,d in config.items():
                    if k in kwargs.keys():
                        config[k] = kwargs[k]
                        added_keywords.append(k)
                    else:
                        recursive_add(kwargs,d)
                        
        recursive_add(kwargs, self.config)
        
        # any non-found keyword in default will be regarded as 
        # keyword for hBayesDM
        for keyword,value in kwargs.items():
            if keyword not in added_keywords:
                self.config['HBAYESDM'][keyword] = value
        
    def _copy_config(self):
        """
        deep copy of configuration dictionary,
        skipping values which are not writable on yaml file.
        """
        def is_writable(d):
            if isinstance(d,str) or \
                isinstance(d, list) or \
                isinstance(d, tuple) or \
                isinstance(d, int) or \
                isinstance(d, float): 
                return True
            else:
                return False
                
        def recursive_copy(config):
            copied = {}
            for k,d in config.items():
                if isinstance(d,dict):
                    copied[k] = recursive_copy(d)
                elif is_writable(d):
                    copied[k] = d
            return(copied)
        
        return recursive_copy(self.config)
    
    def run(self):
        """
        """
        pass