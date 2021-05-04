import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
from mbmvpa.models.mvpa_general import MVPA_Base
import numpy as np
from ..data.tf_generator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tempfile
import random
import os
from pathlib import Path


class MVPA_MLP(MVPA_Base):
    
    def __init__(self, 
                 input_shape,
                 layer_dims=[1024, 1024],
                 activation="linear",
                 activation_output="linear",
                 dropout_rate=0.5,
                 val_ratio=0.2,
                 optimizer="adam",
                 loss="mse",
                 learning_rate=0.001,
                 n_epoch = 50,
                 n_patience = 10,
                 n_batch = 64,
                 n_sample = 30000,
                 use_bias = True,
                 use_bipolar_balancing = False,
                 gpu_visible_devices = None,
                 **kwargs):
        
        self.name = "MLP_TF"
        if isinstance(input_shape, int):
            input_shape = [input_shape,]
        self.input_shape = input_shape
        self.layer_dims = layer_dims
        self.activation = activation
        self.activation_output = activation_output
        self.dropout_rate = dropout_rate
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.use_bias = use_bias
        self.n_patience = n_patience
        self.n_batch = n_batch
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.val_ratio = val_ratio
        self.use_bipolar_balancing = use_bipolar_balancing
        self.model = None
        if gpu_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(v) for v in gpu_visible_devices]v)
    
    
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
        train_generator = DataGenerator(X_train, y_train, self.n_batch, shuffle=True,
                                        use_bipolar_balancing=self.use_bipolar_balancing)
        val_generator = DataGenerator(X_test, y_test, self.n_batch, shuffle=False,
                                        use_bipolar_balancing=self.use_bipolar_balancing)
        
        
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
                      verbose=0, callbacks=[mc, es],
                      validation_data=val_generator,
                      steps_per_epoch=train_steps,
                      validation_steps=val_steps)

        # load best model
        self.model.load_weights(best_model_filepath)
        
        os.remove(best_model_filepath+'.data-00000-of-00001')
        os.remove(best_model_filepath+'.index')
        
        return
    
    def predict(self,X,**kwargs):
        return self.model.predict(X)
    
    def get_weights(self,**kwargs):
        weights = []
        for layer in self.model.layers:
            if "dense" not in layer.name:
                continue
            weights.append(layer.get_weights()[0])

        coef = weights[0]
        for weight in weights[1:]:
            coef = np.matmul(coef, weight)

        return coef
    