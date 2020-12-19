#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#author: Cheol Jun Cho, Yedarm Seong
#contact: cjfwndnsl@gmail.com, mybirth0407@gmail.com
#last modification: 2020.11.03

import datetime
from pathlib import Path


class Model():
    def __init__(self,save_path='.'):
        self.save_path = Path(save_path)
        self.plot_path = None
        self.log_path = None
        
    def _make_log_dir():
        now = datetime.datetime.now()
        save_root = self.save_path / f'report_{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'
        self.plot_path = save_root / 'plot'
        self.log_path = save_root / 'log'
        
        save_root.mkdir()
        self.plot_path.mkdir()
        self.log_path.mkdir()
        
        return
    
    def fit():
        pass
    
    def coeff():
        pass
    
    def _coeff_raw():
        pass
    
        
