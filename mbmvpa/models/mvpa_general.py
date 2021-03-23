#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2021.03.23

import datetime
from pathlib import Path
import numpy as np
from tqdm import tqdm
from bids import BIDSLayout
import matplotlib.pyplot as plt

from ..utils import config


class MVPA_CV():
    
    def __init__(self, 
                X_dict,
                y_dict,
                model,
                model_param_dict={},
                method='5-fold',
                n_cv_repeat=1,
                cv_save=True,
                cv_save_path=".",
                task_name="unnamed",
                report_function_dict=None):
        
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.model = model
        self.model_param_dict = model_param_dict
        self.method = method
        self.n_cv_repeat = n_cv_repeat
        self.cv_save = cv_save
        self.cv_save_path = cv_save_path
        self.task_name = task_name
        self.report_function_dict = report_function_dict
        self.output_names = ['weights','pred_train','pred_test']
        
        if self.cv_save:
            now = datetime.datetime.now()
            self.save_root = Path(cv_save_path) / f'{self.task_name}_{self.method}_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
            self.save_root.mkdir()
            
    
    def _run_singletime(X_train,
                        y_train,
                        X_test,
                        y_test):
        
        self.model.reset()
        self.model.fit(X_train,y_train, **self.model_param_dict)
        pred_train = self.model.predict(y_train)
        pred_test = self.model.predict(y_test)
        if 'get_weights' in dir(self.model):
            weights = self.model.get_weights() # {weight_name: weight_tensor_value}
        else:
            weights = None
            
        output = {'weights':weights,
                 'pred_train':pred_train,
                 'pred_test':pred_test}
        
        if 'report' in dir(self.model):
            additional_reports = self.model.report()
            for name, data in additional_reports.items():
                output[name] = data
                if name not in self.output_names:
                    self.output_names.append(name)
        
        return output
    
    def run():
        
        outputs = {}
        if self.method=='loso':#leave-one-subject-out
            subject_list = list(self.X_dict.keys())
            for j in tqdm(range(n_cv_repeat)):
                for subject_id in tqdm(subject_list):
                    X_test = X_dict[subject_id]
                    y_test = y_dict[subject_id]
                    X_train = np.concatenate([X_dict[v] for v in subject_list if v != subject_id],0)
                    y_train = np.concatenate([y_dict[v] for v in subject_list if v != subject_id],0)
                    outputs[f'{j}-{subject_id}'] = _run_singletime(X_train, y_train, X_test, y_test)
                
        elif 'fold' in self.method:
            n_fold = int(method.split('-')[0])
            X = np.concatenate([v for _,v in X_dict.items()],0)
            y = np.concatenate([v for _,v in y_dict.items()],0)
            for j in tqdm(range(n_cv_repeat)):
                np.random.seed(42+j)
                ids = np.arange(X.shape[0])
                fold_size = X.shape[0]//n_fold
                for i in range(n_fold):
                    test_ids = ids[fold_size*i:fold_size*(i+1)]
                    train_ids = np.concatenate([ids[:fold_size*i],ids[fold_size*(i+1):]],0)
                    X_test = X[test_ids]
                    y_test = y[test_ids]
                    X_train = X[train_ids]
                    y_train = y[train_ids]
                    outputs[f'{j}-{i}'] = _run_singletime(X_train, y_train, X_test, y_test)
                    
        if self.cv_save:
            report_path = self.save_root/ 'raw_result'
            report_path.mkdir()
            for report_id, output in outputs.items():
                for key, data in output.items():
                    np.save(str(report_path / f'{report_id}_{key}.npy'), data)
        
        def reshape_dict(_dict, inner_keys):
            return {inner_key: {key:data[inner_key] for key,data in _dict.items()} for inner_key in inner_keys}
            
        def check_report_key(keys):
            for key in keys:
                assert key in self.output_names
                
        for report_key, function in self.report_function_dict:
            if not isinstance(report_key,list):
                report_key = [report_key]
            check_report_key(keys)
            function(save=self.cv_save,
                     save_path=self.cv_save_path,
                     **reshape_dict(outputs,report_key))
            
        return outputs