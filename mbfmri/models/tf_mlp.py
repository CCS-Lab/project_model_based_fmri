#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
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

from mbfmri.models.tf_base import MVPA_TF
from mbfmri.utils.report import *

tf.get_logger().setLevel("ERROR")

class MVPA_MLP(MVPA_TF):
    
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
                 activation="sigmoid",
                 activation_output=None,
                 dropout_rate=0.5,
                 batch_norm=False,
                 val_ratio=0.2,
                 optimizer="adam",
                 learning_rate=0.001,
                 n_epoch = 50,
                 n_min_epoch = 5,
                 n_patience = 10,
                 n_batch = 64,
                 n_sample = 30000,
                 use_bias = True,
                 logistic = False,
                 explainer = None,
                 train_verbosity=0,
                 model_save_path=None,
                 l1_regularize=0,
                 l2_regularize=0,
                 **kwargs):
        
        super().__init__(val_ratio=val_ratio,
                         optimizer=optimizer,
                         learning_rate=learning_rate,
                         n_epoch = n_epoch,
                         n_min_epoch = n_min_epoch,
                         n_patience = n_patience,
                         n_batch = n_batch,
                         n_sample = n_sample,
                         explainer = explainer,
                         train_verbosity=train_verbosity,
                         model_save_path=model_save_path,)
        
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
        if activation_output is not None:
            self.activation_output = activation_output
            
        self.use_bias = use_bias
        self.model = None
        self.X_train = None
        self.X_test = None
        self.explainer = explainer
        self.batch_norm = batch_norm
        
        self.dense_kwargs = {'use_bias': use_bias,
                            'kernel_regularizer': l1_l2(l1=l1_regularize, 
                                                        l2=l2_regularize)}
        if use_bias:
            self.dense_kwargs['bias_regularizer']=l1_l2(l1=l1_regularize,
                                                        l2=l2_regularize)
        self.dropout_rate = dropout_rate
        
    def reset(self,**kwargs):
        
        self.model = Sequential()
        self.model.add(Dense(self.layer_dims[0],
                        activation=self.activation,
                        input_shape=self.input_shape,
                        **self.dense_kwargs))
    
        if self.batch_norm:
            self.model.add(BatchNormalization())
        if self.dropout_rate > 0:
            self.model.add(Dropout(self.dropout_rate))

        # add layers
        for dim in self.layer_dims[1:]:
            self.model.add(Dense(dim, activation=self.activation, **self.dense_kwargs))
            if self.batch_norm:
                self.model.add(BatchNormalization())
            if self.dropout_rate > 0:
                self.model.add(Dropout(self.dropout_rate))

        self.model.add(Dense(1, activation=self.activation_output, **self.dense_kwargs))
        
        # set optimizer
        if self.optimizer == "adam":
            optlayer = Adam(learning_rate=self.learning_rate,name=self.optimizer)
        else: # not implemented
            optlayer = Adam(learning_rate=self.learning_rate,name=self.optimizer)

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        return 
