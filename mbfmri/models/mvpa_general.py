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
from mbfmri.utils import config
from mbfmri.utils.plot import plot_pearsonr
import random
import nibabel as nib
from scipy.stats import ttest_1samp


class MVPA_Base():
    
    # interface for MVPA model
    # recommend to add **kwargs for each function
    # to avoid any conflict
    
    def __init__(self,
                 voxel_mask=None,
                 **kwargs):
        self.name = "unnamed"
        
    def reset(self,**kwargs):
        pass
    
    def fit(self,X,y,**kwargs):
        pass
    
    def predict(self,X,**kwargs):
        pass
    
class MVPA_CV_H():
    r"""
    
    Hierarchical version of MVPA_CV.
    MVPA_CV_H runs MVPA_CV on each subject to get individual brain maps.(first-level brain maps)
    Then, by one sample T test, first-level brain maps will be converted to a 2nd-level brain map.
    
    Parameters
    ----------
    X_dict : dict
        dictionary containing voxel features for each subject.
        X_dict[{subject_id}] = np.array(time, voxel_num)
    y_dict : dict
        dictionary containing bold-like signals of latent process for each subject.
        y_dict[{subject_id}] = np.array(time,)
    model : MVPA_Base
        MVPA model implemented uppon *MVPA_Base* interface. 
        ElasticNet, MLP and CNN are available.
        please refer to the documentation
    model_param_dict : dict, default={}
        dictionary for keywarded arguments, which will be additionally fed to MVPA model.
        its uppon MVPA model implementation. Please check the codes of the models.
        (not used in models offered by this package.)
    method : str, default='5-fold'
        name of method for cross-validation
        leave-n-subject-out is not allowed as cross-validation will be applied to one subject at a time.
        "{n}-fold" : n will be parsed as integer and n cross-validation will be conducted.
    n_cv_repaet : int, default=1
        number of repetition of running cross-validation
        bigger the valeu, more time for computation and more stability.
    cv_save : boolean, default=True
        indicate whether save results or not
    cv_save_path : str or pathlib.PosixPath, default='.'
        path for saving results.
    experiment_name : str, default="unnamed"
        name for cross-validation experiment
    report_function : dict, default={}
        dictionary for report function. 
        please refer to mbfmri.utils.report for the detail.
    """
    
    def __init__(self, 
                X_dict,
                y_dict,
                model,
                model_param_dict={},
                method='5-fold',
                n_cv_repeat=1,
                cv_save=True,
                cv_save_path=".",
                experiment_name="unnamed",
                reporter=None):
    
        assert len(X_dict) == len(y_dict)
        # leave-n-subject-out is not allowed as MVPA is fitted on one subject at a time.
        assert 'fold' in method 
        
        self.n_subject = len(X_dict)
        
        # initiate MVPA_CV instance for each subject separately
        self.mvpa_cv_dict= {subj_id:MVPA_CV(X_dict={subj_id:X_dict[subj_id]},
                                            y_dict={subj_id:y_dict[subj_id]},
                                            model=model,
                                            model_param_dict=model_param_dict,
                                            method=method,
                                            n_cv_repeat=n_cv_repeat,
                                            cv_save=False,
                                            cv_save_path=None,
                                           experiment_name=experiment_name,
                                           reporter=reporter) for subj_id in X_dict.keys()}
        
        self.experiment_name = experiment_name
        self.cv_save = cv_save
        
        # enter subjectwise save path in each MVPA_CV instance
        if self.cv_save:
            now = datetime.datetime.now()
            self.save_root = Path(cv_save_path) / f'report_{model.name}_{experiment_name}_hierarchical-{method}_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
            self.save_root.mkdir(exist_ok=True)
            for subj_id in X_dict.keys():
                subj_save_root = self.save_root / subj_id
                subj_save_root.mkdir(exist_ok=True)
                self.mvpa_cv_dict[subj_id].save_root = subj_save_root
                self.mvpa_cv_dict[subj_id].cv_save = True
    
    def run(self,
            pval_threshold_2nd=0.01,
            **kwargs):
        
        # run MVPA_CV for each subject
        output_dict = {}
        for subj_id, mvpa_cv in tqdm(self.mvpa_cv_dict.items()):
            print(f"INFO: MVPA_CV on subject-{subj_id}")
            output_dict[subj_id] = mvpa_cv.run(**kwargs)
            
        # get 2nd-level T-map
        self._make_2nd_t_map()
        # plot distribution of pearson r value of each subject.
        self._plot_pearsonr(save=self.cv_save,
                            pval_threshold=pval_threshold_2nd)
        return output_dict
    
    def _make_2nd_t_map(self):
        
        if self.cv_save:
            # aggregate first-level brain maps
            nii_files = [f for f in self.save_root.glob('**/*.nii')]
            if len(nii_files) != 0:
                print(f"INFO: {len(nii_files)} first-level brain images is(are) found.")
                nii_loaded = [nib.load(f) for f in nii_files]
                activation_maps = np.array([f.get_fdata() for f in nii_loaded])
                
                # get (one-sample) T-map from first-level brain maps.
                t_map_2nd = ttest_1samp(activation_maps, 0).statistic
                # save
                nib.Nifti1Image(t_map_2nd,
                                affine=nii_loaded[0].affine).to_filename(self.save_root/f'{self.experiment_name}_second_t_map.nii')
                print("INFO: second-level brain map is created.")
            else:
                print("INFO: No first-level brain image is found.")
        
    
    def _plot_pearsonr(self, 
                       save=True,
                       pval_threshold=0.01):
        
        pred_test_list = []
        y_test_list = []
        pred_train_list = []
        y_train_list = []
        
        def _get_subjects_pred(subject_id):
            pred_test_files = [f for f in self.save_root.glob(f'{subject_id}/**/*pred_test.npy')]
            y_test_files = [f for f in self.save_root.glob(f'{subject_id}/**/*y_test.npy')]
            pred_train_files = [f for f in self.save_root.glob(f'{subject_id}/**/*pred_train.npy')]
            y_train_files = [f for f in self.save_root.glob(f'{subject_id}/**/*y_train.npy')]
            pred_test_files.sort()
            y_test_files.sort()
            pred_train_files.sort()
            y_train_files.sort()

            pred_test = np.concatenate([np.load(f) for f in pred_test_files],0)
            y_test = np.concatenate([np.load(f) for f in y_test_files],0)
            pred_train = np.concatenate([np.load(f) for f in pred_train_files],0)
            y_train = np.concatenate([np.load(f) for f in y_train_files],0)

            return pred_test, y_test, pred_train, y_train
        
        for subject_id in self.mvpa_cv_dict.keys():
            pred_test, y_test,\
                pred_train, y_train= _get_subjects_pred(subject_id)
            pred_test_list.append(pred_test)
            y_test_list.append(y_test)
            pred_train_list.append(pred_train)
            y_train_list.append(y_train)
            
        save_path = self.save_root / 'second_pearsonr'
        save_path.mkdir(exist_ok=True)
        plot_pearsonr(y_train_list,
                      y_test_list,
                      pred_train_list,
                      pred_test_list,
                      save=save,
                      save_path=save_path,
                      pval_threshold=pval_threshold)
            

        
class MVPA_CV():
    
    r"""
    
    MVPA_CV class runs cross-validation experiments for given X, y and a regression model.
    To allow subjec-wise cross-validation, *leave-n-subject-out*,
    X and y data are required to be in dictionary, which a subject's data can be indexed by its ID.
    The cross-validation itself can be repeated, and contain the data involved including data used for training,
    validating, model weights fitted, and additional report values the model offers. 
    (an example of report values is internal cross-validation errors from fitting ElasticNet)
    The data obtained from fitting models then post-processed by report functions in *Reporter*.
    Making a brain activation map from interpreting models is also included in these report functions.
    Please refer to mbfmri.models.elasticnet and mb.models.tf_mlp for detail.
    
    Parameters
    ----------
    X_dict : dict
        dictionary containing voxel features for each subject.
        X_dict[{subject_id}] = np.array(time, voxel_num)
    y_dict : dict
        dictionary containing bold-like signals of latent process for each subject.
        y_dict[{subject_id}] = np.array(time,)
    model : MVPA_Base
        MVPA model implemented uppon *MVPA_Base* interface. 
        ElasticNet, MLP and CNN are available.
        please refer to the documentation
    model_param_dict : dict, default={}
        dictionary for keywarded arguments, which will be additionally fed to MVPA model.
        its uppon MVPA model implementation. Please check the codes of the models.
        (not used in models offered by this package.)
    method : str, default='5-fold'
        name of method for cross-validation
        "{n}-fold" : n will be parsed as integer and n cross-validation will be conducted.
        "{n}-lnso" : n will be parsed as integer and leave-n-subject-out will be conducted.
    n_cv_repaet : int, default=1
        number of repetition of running cross-validation
        bigger the valeu, more time for computation and more stability.
    cv_save : boolean, default=True
        indicate whether save results or not
    cv_save_path : str or pathlib.PosixPath, default='.'
        path for saving results.
    experiment_name : str, default="unnamed"
        name for cross-validation experiment
    report_function : dict, default={}
        dictionary for report function. 
        please refer to mbfmri.utils.report for the detail.
    """
    
    def __init__(self, 
                X_dict,
                y_dict,
                model,
                model_param_dict={},
                method='5-fold',
                n_cv_repeat=1,
                cv_save=True,
                cv_save_path=".",
                experiment_name="unnamed",
                reporter=None
                ):
        
        self.X_dict = X_dict
        self.y_dict = y_dict
        self.model = model
        self.model_param_dict = model_param_dict
        self.method = method
        self.n_cv_repeat = n_cv_repeat
        self.cv_save = cv_save
        self.cv_save_path = cv_save_path
        self.experiment_name = experiment_name
        self.reporter = reporter
        self.output_stats = {}
        
        # set save path with current time
        if self.cv_save:
            now = datetime.datetime.now()
            self.save_root = Path(cv_save_path) / f'report_{self.model.name}_{self.experiment_name}_{self.method}_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
            self.save_root.mkdir(exist_ok=True)
            
    
    def _run_singletime(self,
                        X_train,
                        y_train,
                        X_test,
                        y_test,
                        **kwargs):
        """
        fit an MVPA model and return involved data
        """
        # 1. reset model
        self.model.reset()
        
        # make additional input parameter from model_param_dict
        for k,d in self.model_param_dict.items():
            kwargs[k] = d
            
        # 2. fit model
        self.model.fit(X_train,y_train, **kwargs)
        pred_train = self.model.predict(X_train)
        pred_test = self.model.predict(X_test)
        
        # 3. report results
        
        # if weight extraction is offered, get weights from models
        if 'get_weights' in dir(self.model):
            weights = self.model.get_weights() # {weight_name: weight_tensor_value}
        else:
            weights = None
        
        # basic output layout
        output = {'weights':weights,
                 'pred_train':pred_train,
                  'y_train':y_train,
                  'y_test':y_test,
                 'pred_test':pred_test}
        
        # if additional report function exists,
        # add reported data to the output
        if 'report' in dir(self.model):
            additional_reports = self.model.report(**kwargs)
            for name, data in additional_reports.items():
                output[name] = data
                
        # if save function is implemented in the model, save.
        if 'save' in dir(self.model):
            self.model.save(self.cv_save_path)
        return output
    
    def run(self,**kwargs):
        
        print(f"INFO: start running the experiment. {self.model.name}")
        outputs = {}
        
        if 'lnso' in self.method: #leave-n-subject-out
            
            subject_list = list(self.X_dict.keys())
            n_val_subj = int(self.method.split('-')[0])
            
            assert len(subject_list)>n_val_subj, f"The number of subject must be bigger than the leave-out number, {n_val_subj}."
            
            for j in tqdm(range(self.n_cv_repeat),leave=True, desc='cv_repeat'):
                random.shuffle(subject_list)
                subject_ids_list = [subject_list[i:i + n_val_subj]
                                for i in range(0, len(subject_list), n_val_subj)]
            
                inner_iterater = tqdm(subject_ids_list,desc='subject',leave=False)
                for i, subject_ids in enumerate(inner_iterater):
                    inner_iterater.set_description(f"subject_{subject_ids}")
                    X_test = np.concatenate([self.X_dict[subject_id] for subject_id in subject_ids],0)
                    y_test = np.concatenate([self.y_dict[subject_id] for subject_id in subject_ids],0)
                    
                    X_train = np.concatenate([self.X_dict[v] for v in subject_list if v not in subject_ids],0)
                    y_train = np.concatenate([self.y_dict[v] for v in subject_list if v not in subject_ids],0)
                    kwargs['fold']=i
                    kwargs['repeat']=j
                    outputs[f'{j}-{i}'] = self._run_singletime(X_train, y_train, X_test, y_test, **kwargs)
                
        elif 'fold' in self.method: # n-fold cross-validation
            
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
                    kwargs['fold']=i
                    kwargs['repeat']=j
                    outputs[f'{j}-{i}'] = self._run_singletime(X_train, y_train, X_test, y_test, **kwargs)
        
        # statistics of outputs
        for _,output in outputs.items():
            for key in output.keys():
                if key not in self.output_stats.keys():
                    self.output_stats[key] = 0
                self.output_stats[key] += 1

        print("INFO: output statistics")
        for key, count in self.output_stats.items():
                print(f"      {key:<30}{count}")
        
        # save raw data
        if self.cv_save:
            report_path = self.save_root/ 'raw_result'
            report_path.mkdir(exist_ok=True)
            for report_id, output in outputs.items():
                for key, data in output.items():
                    np.save(str(report_path / f'{report_id}_{key}.npy'), data)
                    
            print(f"INFO: results are saved at {str(report_path)}.")
        
        if self.reporter is not None:
            self.reporter.run(search_path=self.save_root,
                         save=self.cv_save,
                         save_path=self.save_root)
            
        print(f"INFO: running done.")
            
        return outputs