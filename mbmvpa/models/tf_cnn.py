import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, BatchNormalization, Flatten, AveragePooling2D
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

import pdb

class MVPA_CNN(MVPA_Base):
    
    def __init__(self, 
                 input_shape,
                 layer_dims=[8,16,32],
                 kernel_size=[3,3,3],
                 logit_layer_dim=256,
                 activation="relu",
                 activation_output="linear",
                 dropout_rate=0.2,
                 val_ratio=0.2,
                 optimizer="adam",
                 loss="mse",
                 learning_rate=0.001,
                 n_epoch = 50,
                 n_patience = 10,
                 n_batch = 64,
                 n_sample = 30000,
                 batch_norm=True,
                 use_bipolar_balancing = False,
                 voxel_mask=None,
                 gpu_visible_devices = None,
                 **kwargs):
        
        self.name = "CNN_TF"
        self.input_shape = input_shape
        self.layer_dims = layer_dims
        self.kernel_size = kernel_size
        self.logit_layer_dim = logit_layer_dim
        self.activation = activation
        self.activation_output = activation_output
        self.dropout_rate = dropout_rate
        self.batch_norm = batch_norm
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_patience = n_patience
        self.n_batch = n_batch
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.val_ratio = val_ratio
        self.use_bipolar_balancing = use_bipolar_balancing
        self.voxel_mask = voxel_mask
        self.model = None
        if gpu_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(v) for v in gpu_visible_devices])
    
    
    def reset(self,**kwargs):
        self.model = Sequential()
        self.model.add(Conv2D(self.layer_dims[0],
                    (self.kernel_size[0],self.kernel_size[0]),
                    activation=self.activation,
                    padding='same',
                    input_shape=self.input_shape,))
        
        self.model.add(AveragePooling2D(pool_size=(2,2)))
    
        # add layers
        for dim,kernel in zip(self.layer_dims[1:],self.kernel_size[1:]):
            self.model.add(Conv2D(dim,
                        (kernel,kernel),
                        activation=self.activation,
                        padding='same'))
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
    
    def get_weights(self,voxel_mask=None,**kwargs):
        
        outputs_pool = []
        
        if voxel_mask is None:
            voxel_mask = self.voxel_mask
        
        assert voxel_mask is not None, "voxel mask should be given."
        
        sample = []
        for x,y,z in zip(*np.nonzero(self.voxel_mask)):
            temp = np.zeros(self.input_shape) 
            temp[x,y,z] = 1
            sample.append(temp)
        sample = np.array(sample)
        sample_size = (self.voxel_mask == 1).sum()
            
        n_step = int(np.ceil((sample_size+0.0)/self.n_batch))

        outputs = []
        for i in range(n_step):
            output = self.model.predict(sample[i*self.n_batch:(i+1)*self.n_batch])
            output = list(output.flatten())
            outputs += output

        weights = np.array(outputs).ravel()
        
        return weights