#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
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

class MVPA_MLP(MVPA_Base):
    
    r"""
    
    **MVPA_MLP** is an MVPA model implementation of Multi-layer Perceptron (MLP).
    The model is implemented upon Tensorflow (>= 2.0.0).

    Mini-batch gradient descent with earlystopping is adopted for model fitting using Keras APIs.

    Coefficient extraction is done by sequential matrix multiplication of
    layers. The activation function is assumed to be linear.
    Repeat several times (=N) and return N coefficients.
    
    Parameters
    ----------
    
    input_shape : int or [int]
        Dimension of input data, which will be fed as X. 
        It should be same as the number of voxel-feature.
    layer_dims : list of int, default=[1024, 1024]
        List of integer specifying the dimensions of each hidden layer.
        Fully-connected layers will be stacked with the sizes indicated by *layer_dims*.
        The last layer, *layer_dims[-1]* --> *1*, will be added.
    activation : str, default="linear"
        Name of activation function which will be applied to the output of hidden layers.
    activation_output : str, default="linear"
        Name of activation function for the final output.
    dropout_rate : float, default=0.5
        Rate of drop out, which will be applied after the hidden layers.
    val_ratio : float, default=0.2
        Rate for inner cross-validation, which will be used to split input data to 
        (train[1-val_ratio], valid[val_ratio]). The validation dataset will be used for 
        determining *early stopping*.
    optimizer : str, default="adam"
        Name of optimizer used for fitting model. The default optimizer is **Adam**. (https://arxiv.org/abs/1412.6980)
        Please refer to Keras optimizer api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    loss : str, default="mse"
        Name of objective function to minimize in training. as it is a regression, default is 'mse' (Mean Squared Error)
        Please refer to Keras loss api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/losses)
    learning_rate : float, default=0.001
        Tensor, floating point value, or a schedule that is a tf.keras.optimizers.schedules.LearningRateSchedule, or a callable that takes no arguments and returns the actual value to use, The learning rate. Defaults to 0.001.
        Please refer to Keras optimizer api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    n_epoch : int, default=50
        Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
    n_patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
        Please refer to https://keras.io/api/callbacks/early_stopping/
    n_batch : int, default=64
        Number of samples per gradient update.
    n_sample : int, default=30000
        Max number of samples used in a single fitting.
        If the number of data is bigger than *n_samples*, sampling will be done for 
        each model fitting.
        This is for preventing memory overload.
    use_bias : bool, default=True
        If True, bias will be used in layers, otherwise bias term will not be considered.
    gpu_visible_devices : list of str or list of int, default=None
        Users can indicate a list of GPU resources here. 
        It would have a same effect as "CUDA_VSIBLE_DEVICES=..."
    
    """
    
    
    def __init__(self, 
                 input_shape,
                 layer_dims=[1024, 1024],
                 activation="linear",
                 dropout_rate=0.5,
                 val_ratio=0.2,
                 optimizer="adam",
                 learning_rate=0.001,
                 n_epoch = 50,
                 n_patience = 10,
                 n_batch = 64,
                 n_sample = 30000,
                 use_bias = True,
                 gpu_visible_devices = None,
                 logistic = False,
                 explainer = None,
                 train_verbosity=0,
                 **kwargs):
        
        
        self.name = "MLP_TF"
        if isinstance(input_shape, int):
            input_shape = [input_shape,]
        self.input_shape = input_shape
        self.layer_dims = layer_dims
        self.activation = activation
        self.logistic = logistic
        if self.logistic:
            self.activation_output = 'sigmoid'
            self.loss = 'bce'
        else:
            self.activation_output = 'linear'
            self.loss = 'mse'
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.use_bias = use_bias
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
        if gpu_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(v) for v in gpu_visible_devices])
    
    
    def reset(self,**kwargs):
        
        self.model = Sequential()
        self.model.add(Dense(self.layer_dims[0],
                        activation=self.activation,
                        input_shape=self.input_shape,
                        use_bias=self.use_bias))
        self.model.add(Dropout(self.dropout_rate))

        # add layers
        for dim in self.layer_dims[1:]:
            self.model.add(Dense(dim, activation=self.activation, use_bias=self.use_bias))
            self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(1, activation=self.activation_output, use_bias=self.use_bias))
        
        # set optimizer
        if self.optimizer == "adam":
            optlayer = Adam(learning_rate=self.learning_rate,name=self.optimizer)
        else: # not implemented
            optlayer = Adam(learning_rate=self.learning_rate,name=self.optimizer)

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        return 

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
        
        temp = tempfile.NamedTemporaryFile()
        best_model_filepath = temp.name
        
        mc = ModelCheckpoint(
            best_model_filepath,
            save_best_only=True, save_weights_only=True,
            monitor="val_loss", mode="min")

        # device for early stopping. if val_loss does not decrease within patience, 
        # the training will stop
        es = EarlyStopping(monitor="val_loss", patience=self.n_patience)
        
        self.model.fit(train_generator, epochs=self.n_epoch,
                      verbose=self.train_verbosity,
                       callbacks=[mc, es],
                      validation_data=val_generator,
                      steps_per_epoch=train_steps,
                      validation_steps=val_steps)

        # load best model
        self.model.load_weights(best_model_filepath)
        
        os.remove(best_model_filepath+'.data-00000-of-00001')
        os.remove(best_model_filepath+'.index')
        
        self.X_train, self.X_test = X_train,X_test
        
        return
    
    def predict(self,X,**kwargs):
        return self.model.predict(X)
    
    def get_weights(self,**kwargs):
        if self.explainer is not None:
            return self.explainer(self.model,self.X_train,self.X_test)
        else:
            return None
    