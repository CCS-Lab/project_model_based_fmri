#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#author: Cheol Jun Cho, Yedarm Seong
#contact: cjfwndnsl@gmail.com, mybirth0407@gmail.com
#last modification: 2020.11.03

import datetime
from pathlib import Path
import numpy as np

from bids import BIDSLayout
import matplotlib.pyplot as plt

from ..utils import config
from ..utils.coef2map import get_map
from ..models.tensorflow_utils import ExperimenterTF
from ..models.extractor import DefaultExtractor
import pdb
        
    
    
class MVPA_TF():
    ''' Model for MVPA regression
    
    Model abstraction for  MVPA regression model. 
    The implemented model should have 
    
    Attributes:
        X (numpy.ndarray): preprocessed fMRI data. shape : data # x voxel #
        y (numpy.ndarray): parametric modulation values to regress X against. shape: data #
        model (tf.model) : regression model for MVPA
        layout (bids.BIDSLayout): BIDSLayout by bids package. used to get default save_path.
        save_path (Path): root path for saving fitting info. default would be current working dir. 
        plot_path (Path): path for saving plot. it will be *save_path/plot*
        log_path (Path): path for saving fitting log. it wiil be *save_path/log*
        save (bool): if true save all intermittent results.# TODO. not used for now.  
        verbose (int): if > 0 then log fitting process and report a validation mse of each repetition. #TODO add more options
        n_repeat (int): the number of repetition for training models. you will get *n_repeat* number of coefficients.
        n_sample (int): maximum number of instance of data (X,y) used in a single repetition. 
        
    Methods:
        run (callable): method for fitting model with data (X,y) and return coefficients
        
    '''
    
    def __init__(self,
                 X=None,
                 y=None,
                 voxel_mask=None,
                 model=None,
                 model_name=None,
                 extractor=None,
                 extract_n_sample=1,
                 root=None,
                 layout=None,
                 save_path=None,
                 save=True,
                 verbose=1,
                 n_repeat=5,
                 n_sample=10000,
                 n_epoch=100,
                 n_patience=10,
                 n_batch=64,
                 validation_split_ratio=0.2,
                 save_pred=True,
                 use_bipolar_balancing=False
                 ):
        
        if root is not None:
            layout = BIDSLayout(root, derivatives=True)
            
        if save_path is None:
            if layout is None:
                self.save_path = Path('.')
            else:
                sp = Path(
                    layout.derivatives[config.MBMVPA_PIPELINE_NAME].root)\
                    / config.DEFAULT_SAVE_PATH_CKPT 
                if not sp.exists():
                    sp.mkdir()
                sp = sp / model_name
                if not sp.exists():
                    sp.mkdir()
                self.save_path = sp
        else:
            self.save_path = Path(save_path)
            
        if ( X is None or y is None ) and layout is not None:
            loader = BIDSDataLoader(layout=layout)
            X,y = loader.get_total_data()
            voxel_mask = loader.get_voxel_mask()
        
        self.layout=layout
        self.X = X
        self.y = y
        self.voxel_mask = voxel_mask
        self.model = model
        self.model_name = model_name
        if extractor is None:
            self.extractor = DefaultExtractor(X.shape[-1],extract_n_sample)
        else:
            self.extractor = extractor
            
        self.chk_path = None
        self.log_path = None
        self.result_path = None
        self.save = save
        self.verbose = verbose
        self.n_repeat = n_repeat
        self.n_sample = n_sample
        
        self.n_epoch = n_epoch
        self.n_patience = n_patience
        self.n_batch = n_batch
        self.validation_split_ratio = validation_split_ratio
        self.save_pred = save_pred
        self.use_bipolar_balancing = use_bipolar_balancing
        
        self._coeffs = []
        self._errors = []
        self._sham_errors = []
        self._sham_errors = []
        self._make_log_dir()
        self._time = datetime.datetime.now()
        self.experimenter = ExperimenterTF(chk_path=self.chk_path,
                                             result_path=self.result_path,
                                             n_sample=self.n_sample,
                                             n_epoch=self.n_epoch,
                                             n_patience=self.n_patience,
                                             n_batch=self.n_batch,
                                             validation_split_ratio=self.validation_split_ratio,
                                             save_pred=self.save_pred,
                                             use_bipolar_balancing=self.use_bipolar_balancing,
                                             verbose=self.verbose)
        
    def _make_log_dir(self):
        now = datetime.datetime.now()
        save_root = self.save_path / f'report_{now.year}-{now.month:02}-{now.day:02}-{now.hour:02}-{now.minute:02}-{now.second:02}'
        self.chk_path = save_root / 'chekpoint'
        self.log_path = save_root / 'log'
        self.result_path = save_root / 'result'
        
        save_root.mkdir()
        self.chk_path.mkdir()
        self.log_path.mkdir()
        self.result_path.mkdir()
        
        return
    
    def _reset_model(self):
        model = self.model
        for layer in model.layers:
            if hasattr(layer, 'init'):
                init = getattr(layer, 'init')
                new_weights = init(layer.get_weights()[0].shape).get_value()
                bias = shared_zeros(layer.get_weights()[1].shape).get_value()
                layer.set_weights([new_weights, bias])
        model.reset_states()
        
        return model
    
    def run(self):
        
        self._coeffs = []
        self._errors = []
        
        for i in range(1, self.n_repeat + 1): 
            model = self._reset_model()
            model,error = self.experimenter(model, i, self.X, self.y)
            self._errors.append(error)
            coeff = self.extractor(model)
            self._coeffs.append(coeff)

        self._coeffs = np.array(self._coeffs)
        self._errors = np.array(self._errors)
        self._time = datetime.datetime.now()
        
        return self._coeffs
    
    def sham(self):
        
        self._sham_errors = []
        ids = np.arange(len(self.y))
        for i in range(1, self.n_repeat + 1):
            model = self._reset_model()
            np.random.shuffle(ids)
            model,error = self.experimenter(model, i, self.X, self.y[ids])
            self._sham_errors.append(error)
            
        self._sham_errors = np.array(self._sham_errors)
        
        return self._sham_errors
    
    def image(self, voxel_mask=None, task_name=None,
                map_type="z", save_path=None, sigma=1):
        
        assert self._coeffs is not None, (
            "Model fitting is not conducted")
        
        if voxel_mask is None:
            voxel_mask = self.voxel_mask
        if task_name is None or task_name == "":
            try:
                task_name = layout.get_task()[0]
            except:
                task_name = "unnamed_task"
        if save_path is None:
            save_path =self.result_path
            
        task_name = f"{task_name}-{self._time.day:02}-{self._time.hour:02}-{self._time.minute:02}-{self._time.second:02}"
        
        return get_map(self._coeffs, voxel_mask, task_name,
                    map_type=map_type, save_path=save_path, sigma=sigma)
        
        
