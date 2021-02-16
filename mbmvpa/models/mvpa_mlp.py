import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2

from ..models.mvpa_base import MVPA_TF

def build_mlp(input_shape,
             layer_dims=[1024, 1024],
             activation="linear",
             activation_output="linear",
             dropout_rate=0.5,
             optimizer="adam",
             loss="mse"):

    model = Sequential()
    model.add(Dense(layer_dims[0],
                    activation=activation,
                    input_shape=input_shape,
                    use_bias=False))
    model.add(Dropout(dropout_rate))

    # add layers
    for dim in layer_dims[1:]:
        model.add(Dense(dim, activation=activation, use_bias=True))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation=activation_output, use_bias=True))
    model.compile(loss=loss, optimizer=optimizer)

    return model



def extractor_mlp(model):

    weights = []
    for layer in model.layers:
        if "dense" not in layer.name:
            continue
        weights.append(layer.get_weights()[0])

    coef = weights[0]
    for weight in weights[1:]:
        coef = np.matmul(coef, weight)
        
    return coef

class MLP(MVPA_TF):
    
    def __init__(self, 
                 layer_dims=[1024, 1024],
                 activation="linear",
                 activation_output="linear",
                 dropout_rate=0.5,
                 optimizer="adam",
                 loss="mse",
                 use_default_extractor=False,
                 **kwargs):
        
        super(MLP, self).__init__(model_name="MLP",**kwargs)
        input_shape = self.X.shape[1:] 
        self.model = build_mlp(input_shape=input_shape,
                             layer_dims=layer_dims,
                             activation=activation,
                             activation_output=activation_output,
                             dropout_rate=dropout_rate,
                             optimizer=optimizer,
                             loss=loss)
        
        self.layer_dims = layer_dims
        self.input_shape = input_shape
        self.activation = activation
        self.activation_output = activation_output
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        
        if not use_default_extractor:
            self.extractor = extractor_mlp
    
    def _reset_model(self):
        self.model = build_mlp(input_shape=self.input_shape,
                             layer_dims=self.layer_dims,
                             activation=self.activation,
                             activation_output=self.activation_output,
                             dropout_rate=self.dropout_rate,
                             optimizer=self.optimizer,
                             loss=self.loss)