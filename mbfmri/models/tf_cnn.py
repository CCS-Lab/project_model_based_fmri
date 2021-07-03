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

from mbfmri.models.mvpa_general import MVPA_Base, MVPA_CV
from mbfmri.utils.report import *

tf.get_logger().setLevel("ERROR")

class MVPA_CNN(MVPA_Base):
    
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
        It should be same as the shape of voxel mask image.
    method : str, default='5-fold'
        Name for type of cross-validation to use. 
        Currently, two options are available.
            - "N-fold" : *N*-fold cross-valiidation
            - "N-lnso" : leave-*N*-subjects-out
            
        If the "N" should be a positive integer and it will be parsed from the input string. 
        In the case of lnso, N should be >= 1 and <= total subject # -1.
    n_cv_repeat : int, default=1
        Number of repetition of the entire cross-validation.
        Larger the number, (normally) more stable results and more time required.
    cv_save : bool, default=True
        indictates save results or not
    cv_save_path : str or pathlib.PosixPath, default="."
        Path for saving results
    experiment_name : str, default="unnamed"
        Name for a single run of this analysis
        It will be included in the name of the report folder created.
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
    activation_output : str, default="linear"
        Name of activation function for the final output.
    dropout_rate : float, default=0.5
        Rate of drop out, which will be applied after the hidden layers.
    val_ratio : float, default=0.2
        Rate for inner cross-validation, which will be used to split input data to 
        (train[1-val_ratio], valid[val_ratio]). The validation dataset will be used for 
        determining *early stopping*.
    optimizer : str, default="adam"
        Name of optimizer used for fitting model
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
    batch_norm : bool, default=True
        If True, BatchNormalization layer will follow each convolutional layer.
    gpu_visible_devices : list of str or list of int, default=None
        Users can indicate a list of GPU resources here. 
        It would have a same effect as "CUDA_VSIBLE_DEVICES=..."
    """
    
    def __init__(self, 
                 input_shape,
                 layer_dims=[8,16,32],
                 kernel_size=[3,3,3],
                 logit_layer_dim=256,
                 activation="relu",
                 dropout_rate=0.2,
                 val_ratio=0.2,
                 optimizer="adam",
                 learning_rate=0.001,
                 n_epoch = 50,
                 n_min_epoch = 5,
                 n_patience = 10,
                 n_batch = 64,
                 n_sample = 30000,
                 l1_regularize=1e-5,
                 l2_regularize=1e-4,
                 batch_norm=False,
                 logistic=False,
                 explainer=None,
                 train_verbosity=0,
                 **kwargs):
        
        
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
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
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