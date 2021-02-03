import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2

from ..models.mvpa_base import MVPA_TF

def make_custom_model_cnn_2D(fmri_shape):
    
    model = Sequential()
    model.add(Conv2D(8, (3,3), kernel_initializer='he_normal', padding='same', input_shape=fmri_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Conv2D(16, (3,3), kernel_initializer='he_normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(32, (3,3), kernel_initializer='he_normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64, (3,3), kernel_initializer='he_normal', padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))


    model.add(Flatten()) 
    model.add(Dense(128, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    
    model.add(Dense(2, kernel_initializer='he_normal'))
    model.add(Activation('linear'))
    
    model.add(Activation('softmax'))
    
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    return model

def build_cnn(input_shape,
             layer_dims=[8,16,32,64],
             kernel_size=[3,3,3,3],
             logit_layer_dim=128,
             activation="relu",
             activation_output="linear",
             dropout_rate=0.2,
             optimizer="adam",
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
                  
    model.add(Flatten()) 
    model.add(Dense(logit_layer_dim, activation=activation))
    
    model.add(Dense(2, kernel_initializer='he_normal'))
    model.add(Activation('linear'))

    model.add(Dense(1, activation=activation_output))
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
                 loss="mse",
                 **kwargs):
        
        super(CNN, self).__init__(model_name="MLP",**kwargs)
        input_shape = self.X.shape[1:] 
        self.model = build_cnn(input_shape=input_shape,
                             layer_dims=layer_dims,
                             kernel_size=kernel_size,
                             logit_layer_dim=logit_layer_dim,
                             activation=activation,
                             activation_output=activation_output,
                             dropout_rate=dropout_rate,
                             optimizer=optimizer,
                             loss=loss)