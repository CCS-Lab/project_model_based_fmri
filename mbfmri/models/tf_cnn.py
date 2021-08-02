#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#author: Cheol Jun Cho
#contact: cjfwndnsl@gmail.com
#last modification: 2020.06.21

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Flatten, AveragePooling2D
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
import shap

from mbfmri.models.tf_base import MVPA_TF
from mbfmri.utils.report import *

tf.get_logger().setLevel("ERROR")

class MVPA_CNN(MVPA_TF):
    
    r"""
    
    **MVPA_CNN** is an MVPA model implementation of Covolutional Neural Network (CNN).
    The model is implemented upon Tensorflow (>= 2.0.0).
    
    Coefficient extraction is done by reading outputs when feeding identity matrix with size of input dimension. 
    This is upon a rough assumption that the model is linear, so that each row of identity matrix can serve as a probe.
    As the implemented CNN model is non-linear, but the trend might be consistent.
    The activation function is assumed to be linear.
    Repeat several times (=N) and return N coefficients.
    
    Parameters
    ----------
    
    input_shape : tuple of int
        Dimension of input data, which will be fed as X. 

    layer_dims : list of int, default=[8, 16, 32]
        List of integer specifying the dimensions (channels) of each hidden layer.
        Convolutional layers will be stacked with the channel sizes indicated by *layer_dims*.
    
    kernel_size : list of int, default=[3, 3, 3]
        List of integer specifying the kernel size  of each convolutional layer.
    
    logit_layer_dim : int, default=256
        Size of a Fully-connected layer, which will be added on convolutional layers.
        The last layer, *logit_layer_dim* --> *1*, will be added for regression.
    
    activation : str, default="linear"
        Name of activation function which will be applied to the output of hidden layers.
    
    activation_output : str, default=None
        Name of activation function for the final output.
        If None (default), it will be automatically determined as 
        "linear" for linear regression and
        "sigmoid" for logistic regression.

    use_bias : bool, default=True
        Indicate if bias is required. 
        If True, bias will be used in layers, otherwise bias term will not be considered.
    
    dropout_rate : float, default=0.5
        Rate of drop out, which will be applied after the last logit layers.

    batch_norm : bool, default=False
        Indicate if batch normalization (BN) is applied.
        If True, BN will be done after each layer before the dropout layer.

    logistic : bool, default=False
        Indicate if logistic regression is required.
        If True, the input should be binary and binary classification model
        will be trained.

    l1_regularize : float, default=0
        Value for L1 penalty of all the weights in the model.

    l2_regularize : float, default=0
        Value for L2 penalty of all the weights in the model.

    val_ratio : float, default=0.2
        Rate for inner cross-validation, which will be used to split input data to 
        (train[1-val_ratio], valid[val_ratio]). The validation dataset will be used for 
        determining *early stopping*.

    optimizer : str, default="adam"
        Name of optimizer used for fitting model. The default optimizer is **Adam**. (https://arxiv.org/abs/1412.6980)
        Please refer to Keras optimizer api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    
    loss : str, default=None,
        Name of objective function to minimize in training. 
        If None (default), it will be automatically determined as 
        "mse" (mean squared error) for linear regression and
        "bce" (binary cross entropy) for logistic regression.
        Please refer to Keras loss api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/losses)

    learning_rate : float, default=0.001
        Tensor, floating point value, or a schedule that is a tf.keras.optimizers.schedules.LearningRateSchedule, or a callable that takes no arguments and returns the actual value to use, The learning rate. Defaults to 0.001.
        Please refer to Keras optimizer api to use another. (https://www.tensorflow.org/api_docs/python/tf/keras/optimizers)
    
    n_batch : int, default=64
        Number of samples per gradient update.

    n_epoch : int, default=50
        Number of epochs to train the model. An epoch is an iteration over the entire x and y data provided. Note that in conjunction with initial_epoch, epochs is to be understood as "final epoch". The model is not trained for a number of iterations given by epochs, but merely until the epoch of index epochs is reached.
    
    n_min_epoch : int, default=50
        Number of minimum epochs to train the model before applying early stopping.

    n_patience : int, default=10
        Number of epochs with no improvement after which training will be stopped.
        Please refer to https://keras.io/api/callbacks/early_stopping/

    n_sample : int, default=30000
        Max number of samples used in a single fitting.
        If the number of data is bigger than *n_samples*, sampling will be done for 
        each model fitting.
        This is for preventing memory overload.

    train_verbosity : int, default=0
        Level of verbosity for model fitting. If it is 1, the reports from
        keras model fitting will be printed.

    model_save_path : str or pathlib.PosixPath, default=None
        Path for saving best models. If not given, trained models will not be saved.

    explainer : utils.explain.Explainer
        Explainer object for interpeting the trained models.
        Please refer to the ``explainer`` module.

    """
    
    def __init__(self, 
                 input_shape,
                 layer_dims=[8,16,32],
                 kernel_size=[3,3,3],
                 logit_layer_dim=256,
                 activation="relu",
                 activation_output=None,
                 dropout_rate=0.2,
                 batch_norm=True,
                 logistic=False,
                 l1_regularize=0,
                 l2_regularize=0,
                 val_ratio=0.2,
                 optimizer="adam",
                 loss=None,
                 learning_rate=0.001,
                 n_batch = 64,
                 n_epoch = 50,
                 n_min_epoch = 5,
                 n_patience = 10,
                 n_sample = 30000,
                 train_verbosity=0,
                 model_save_path=None,
                 explainer=None,
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
        
        
        self.name = "CNN_TF"
        self.input_shape = input_shape
        self.layer_dims = layer_dims
        self.kernel_size = kernel_size
        self.logit_layer_dim = logit_layer_dim
        self.activation = activation
        self.logistic = logistic
        if self.logistic:
            self.activation_output = 'sigmoid'
            self.loss = 'bce'
        else:
            self.activation_output = 'linear'
            self.loss = 'mse'

        # overrided by user input.
        if activation_output is not None:
            self.activation_output = activation_output

        if loss is not None:
            self.loss = loss
        
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.model = None
        self.X_train = None
        self.X_test = None
        self.explainer = explainer
        self.train_verbosity = train_verbosity
        self.conv_kwargs = {'kernel_regularizer': l1_l2(l1=l1_regularize, 
                                                        l2=l2_regularize),
                            'bias_regularizer': l1_l2(l1=l1_regularize, 
                                                      l2=l2_regularize)}

    
    def reset(self,**kwargs):
        self.model = Sequential()
        self.model.add(Conv2D(self.layer_dims[0],
                    (self.kernel_size[0],self.kernel_size[0]),
                    activation=self.activation,
                    padding='same',
                    input_shape=self.input_shape,
                             **self.conv_kwargs))
        
        self.model.add(AveragePooling2D(pool_size=(2,2)))
    
        # add layers
        for dim,kernel in zip(self.layer_dims[1:],self.kernel_size[1:]):
            self.model.add(Conv2D(dim,
                        (kernel,kernel),
                        activation=self.activation,
                        padding='same',
                                 **self.conv_kwargs))
            self.model.add(AveragePooling2D(pool_size=(2,2)))
            
            if self.batch_norm:
                self.model.add(BatchNormalization())
                  
        self.model.add(Flatten()) 
        self.model.add(Dense(self.logit_layer_dim, activation=self.activation))
        self.model.add(Dense(1, activation=self.activation_output))
    

        if self.optimizer == "adam":
            optlayer = Adam(learning_rate=self.learning_rate,name=self.optimizer)
        else: # not implemented
            optlayer = Adam(learning_rate=self.learning_rate,name=self.optimizer)

        self.model.compile(loss=self.loss, optimizer=self.optimizer)

        return 
