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
from mbmvpa.utils import config

import pdb

class MVPA_Base():
    def __init__(self):
        self.name = "unnamed"
    def reset(self):
        return
    def fit(self,X,y):
        return
    def predict(self,X):
        return 
    
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
                report_function_dict={}):
        
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
        self.output_stats = {}
        
        if self.cv_save:
            now = datetime.datetime.now()
            self.save_root = Path(cv_save_path) / f'report_{self.model.name}_{self.task_name}_{self.method}_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
            self.save_root.mkdir()
            
    
    def _run_singletime(self,
                        X_train,
                        y_train,
                        X_test,
                        y_test):
        
        self.model.reset()
        self.model.fit(X_train,y_train, **self.model_param_dict)
        pred_train = self.model.predict(X_train)
        pred_test = self.model.predict(X_test)
        if 'get_weights' in dir(self.model):
            weights = self.model.get_weights() # {weight_name: weight_tensor_value}
        else:
            weights = None
            
        output = {'weights':weights,
                 'pred_train':pred_train,
                  'y_train':y_train,
                  'y_test':y_test,
                 'pred_test':pred_test}
        
        if 'report' in dir(self.model):
            additional_reports = self.model.report()
            for name, data in additional_reports.items():
                output[name] = data
        if 'save' in dir(self.model):
            self.model.save(self.cv_save_path)
        return output
    
    def run(self):
        
        print(f"INFO: start running the experiment. {self.model.name}")
        outputs = {}
        if self.method=='loso':#leave-one-subject-out
            subject_list = list(self.X_dict.keys())
            assert len(subject_list)>1, "The number of subject must be bigger than one."
            for j in tqdm(range(self.n_cv_repeat),leave=True, desc='cv_repeat'):
                inner_iterater = tqdm(subject_list,desc='subject',leave=False)
                for subject_id in inner_iterater:
                    inner_iterater.set_description(f"subject_{subject_id}")
                    X_test = self.X_dict[subject_id]
                    y_test = self.y_dict[subject_id]
                    X_train = np.concatenate([self.X_dict[v] for v in subject_list if v != subject_id],0)
                    y_train = np.concatenate([self.y_dict[v] for v in subject_list if v != subject_id],0)
                    outputs[f'{j}-{subject_id}'] = self._run_singletime(X_train, y_train, X_test, y_test)
                
        elif 'fold' in self.method:
            n_fold = int(self.method.split('-')[0])
            X = np.concatenate([v for _,v in self.X_dict.items()],0)
            y = np.concatenate([v for _,v in self.y_dict.items()],0)
            for j in tqdm(range(self.n_cv_repeat),leave=True,desc='cv_repeat'):
                np.random.seed(42+j)
                ids = np.arange(X.shape[0])
                fold_size = X.shape[0]//n_fold
                inner_iterater = tqdm(range(n_fold),desc='fold',leave=False)
                for i in inner_iterater:
                    inner_iterater.set_description(f"fold_{i+1}")
                    test_ids = ids[fold_size*i:fold_size*(i+1)]
                    train_ids = np.concatenate([ids[:fold_size*i],ids[fold_size*(i+1):]],0)
                    X_test = X[test_ids]
                    y_test = y[test_ids]
                    X_train = X[train_ids]
                    y_train = y[train_ids]
                    outputs[f'{j}-{i}'] = self._run_singletime(X_train, y_train, X_test, y_test)
                 
        for _,output in outputs.items():
            for key in output.keys():
                if key not in self.output_stats.keys():
                    self.output_stats[key] = 0
                self.output_stats[key] += 1
            
        print("INFO: output statistics")
        for key, count in self.output_stats.items():
                print(f"      {key:<30}{count}")
            
        if self.cv_save:
            report_path = self.save_root/ 'raw_result'
            report_path.mkdir()
            for report_id, output in outputs.items():
                for key, data in output.items():
                    np.save(str(report_path / f'{report_id}_{key}.npy'), data)
                    
            print(f"INFO: results are saved at {str(report_path)}.")
        
        def reshape_dict(_dict, inner_keys):
            return {inner_key: {key:data[inner_key] for key,data in _dict.items()} for inner_key in inner_keys}
            
        def check_report_key(keys):
            for key in keys:
                assert key in self.output_stats.keys(), f'{key} is not in {str(list(self.output_stats.keys()))}'
                
        for report_key, function in self.report_function_dict.items():
            
            report_name, report_key = report_key[0], report_key[1:]
            check_report_key(report_key)
            save_path = Path(self.save_root)/report_name 
            save_path.mkdir(exist_ok=True)
            function(save=self.cv_save,
                     save_path=save_path,
                     **reshape_dict(outputs,report_key))
            
        print(f"INFO: {len(self.report_function_dict)} report(s) is(are) done.")
        print(f"INFO: running done.")
            
        return outputs