#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from mbfmri.data.tf_generator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tempfile
import random
import os
from pathlib import Path

from mbfmri.models.mvpa_general import MVPA_Base, MVPA_CV
from mbfmri.utils.report import *

tf.get_logger().setLevel("ERROR")

class MVPA_TF(MVPA_Base):
    
    r"""
    
    """
    
    def __init__(self,
                 val_ratio=0.2,
                 optimizer="adam",
                 learning_rate=0.001,
                 n_epoch = 50,
                 n_min_epoch = 5,
                 n_patience = 10,
                 n_batch = 64,
                 n_sample = 30000,
                 explainer = None,
                 train_verbosity=0,
                 model_save_path=None,
                 **kwargs):
        
        
        self.name = None
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.n_min_epoch = n_min_epoch
        self.n_patience = n_patience
        self.n_batch = n_batch
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.val_ratio = val_ratio
        self.model = None
        self.X_train = None
        self.X_test = None
        self.explainer = explainer
        self.train_verbosity = train_verbosity
        
    def reset(self,**kwargs):
        pass
    

    def fit(self,X,y,**kwargs):
        # add saving total weights. get input from user
        if self.model is None:
            self.reset()
            
        ids = np.arange(X.shape[0])

        if X.shape[0] > self.n_sample:
            np.random.shuffle(ids)
            ids = ids[:self.n_sample]

        # split data to training set and validation set
        train_ids, test_ids = train_test_split(
            ids, test_size=self.val_ratio
        )
        train_steps = len(train_ids) // self.n_batch
        val_steps = len(test_ids) // self.n_batch

        assert train_steps > 0
        assert val_steps > 0

        X_train, X_test = X[train_ids], X[test_ids]
        y_train, y_test = y[train_ids], y[test_ids]
        
        # create helper class for generating data
        # support mini-batch training implemented in Keras
        train_generator = DataGenerator(X_train, y_train, self.n_batch, shuffle=True)
        val_generator = DataGenerator(X_test, y_test, self.n_batch, shuffle=False)
        
        
        #best_model_filepath = tempdir + f"/{self.name}_best_{int(random.random()*100000)}.ckpt"
        if 'model_save_path' in kwargs.keys() and kwargs['model_save_path'] is None:
            use_tempfile = True
            temp = tempfile.NamedTemporaryFile()
            best_model_filepath = temp.name
        else:
            use_tempfile = False
            if 'repeat' in kwargs.keys() and 'fold' in kwargs.keys():
                best_model_filepath = str(kwargs['model_save_path'])+ f"/{kwargs['repeat']}-{kwargs['fold']}_best"
            else:
                best_model_filepath = str(kwargs['model_save_path'])+ f"/{int(random.random()*100000)}-{int(random.random()*100000)}_best"
    
        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min")

        # device for early stopping. if val_loss does not decrease within patience, 
        # the training will stop
        es = EarlyStopping(monitor="val_loss", patience=self.n_patience)
        
        self.model.fit(train_generator, epochs=self.n_min_epoch,
                      verbose=self.train_verbosity,
                      validation_data=val_generator,
                      steps_per_epoch=train_steps,
                      validation_steps=val_steps)
        
        self.model.fit(train_generator, epochs=self.n_epoch-self.n_min_epoch,
                      verbose=self.train_verbosity,
                       callbacks=[mc, es],
                      validation_data=val_generator,
                      steps_per_epoch=train_steps,
                      validation_steps=val_steps)

        # load best model
        self.model.load_weights(best_model_filepath)
        
        if use_tempfile:
            os.remove(best_model_filepath+'.data-00000-of-00001')
            os.remove(best_model_filepath+'.index')
        
        self.X_train, self.X_test = X_train,X_test
        
        return
    
    def predict(self,X,**kwargs):
        return self.model.predict(X)
    
    def get_weights(self,**kwargs):
        if self.explainer is not None:
            return self.explainer(self.model,self.X_test,self.X_train)
        else:
            return None
    