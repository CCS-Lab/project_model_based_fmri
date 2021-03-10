import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l1_l2

from ..models.mvpa_base import MVPA_TF

def build_cnn(input_shape,
             layer_dims=[8,16,32],
             kernel_size=[3,3,3],
             logit_layer_dim=256,
             activation="relu",
             activation_output="linear",
             dropout_rate=0.2,
             batch_norm=False,
             optimizer="adam",
             learning_rate=0.01,
             loss="mse"):
    

    model = Sequential()
    model.add(Conv2D(layer_dims[0],
                    (kernel_size[0],kernel_size[0]),
                    activation=activation,
                    padding='same',
                    input_shape=input_shape,))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    # add layers
    for dim,kernel in zip(layer_dims[1:],kernel_size[1:]):
        model.add(Conv2D(dim,
                    (kernel,kernel),
                    activation=activation,
                    padding='same',
                    input_shape=input_shape,))
        model.add(MaxPooling2D(pool_size=(2,2)))
        if batch_norm:
            model.add(BatchNormalization())
                  
    model.add(Flatten()) 
    model.add(Dense(logit_layer_dim, activation=activation))
    
    model.add(Dense(2, kernel_initializer='he_normal'))
    model.add(Activation('linear'))

    model.add(Dense(1, activation=activation_output))
    
    if optimizer == "adam":
        optlayer = Adam(learning_rate=learning_rate,name=optimizer)
    else:
        optlayer = Adam(learning_rate=learning_rate,name=optimizer)
        
    model.compile(loss=loss, optimizer=optimizer)

    return model



class CNN(MVPA_TF):
    
    def __init__(self, 
                 layer_dims=[8,16,32,64],
                 kernel_size=[3,3,3,3],
                 logit_layer_dim=128,
                 activation="relu",
                 activation_output="linear",
                 dropout_rate=0.15,
                 optimizer="adam",
                 learning_rate=0.001,
                 loss="mse",
                 **kwargs):
        
        super(CNN, self).__init__(model_name="MLP",**kwargs)
        input_shape = self.X.shape[1:] 
        
        self.input_shape = input_shape
        self.layer_dims = layer_dims
        self.kernel_size = kernel_size
        self.logit_layer_dim = logit_layer_dim
        self.activation = activation
        self.activation_output = activation_output
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.model = build_cnn(input_shape=self.input_shape,
                             layer_dims=self.layer_dims,
                             kernel_size=self.kernel_size,
                             logit_layer_dim=self.logit_layer_dim,
                             activation=self.activation,
                             activation_output=self.activation_output,
                             dropout_rate=self.dropout_rate,
                             optimizer=self.optimizer,
                             learning_rate=self.learning_rate,
                             loss=self.loss)
        
    def _reset_model(self):
        self.model = build_cnn(input_shape=self.input_shape,
                             layer_dims=self.layer_dims,
                             kernel_size=self.kernel_size,
                             logit_layer_dim=self.logit_layer_dim,
                             activation=self.activation,
                             activation_output=self.activation_output,
                             dropout_rate=self.dropout_rate,
                             optimizer=self.optimizer,
                             loss=self.loss)