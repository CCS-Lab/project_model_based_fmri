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
from .tf_mlp import MVPA_MLP


    
class MVPA_MLP_SHAP(MVPA_MLP):
    
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
                 use_null_background = False,
                 background_num = 1000,
                 sample_num = 1000
                 **kwargs):
        
        super().__init__(input_shape,
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
                         **kwargs)
        
        self.background_num = background_num
        self.use_null_background = use_null_background
        self.sample_num = sample_num
        self.shap_values = None
        
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
        
        background = X_train[np.random.choice(X_train.shape[0], self.background_num, replace=False)]
        sample =  X_test[np.random.choice(X_test.shape[0], self.sample_num, replace=False)]
        e = shap.DeepExplainer(self.model, background)
        self.shap_values = e.shap_values(sample)[0]
        return
    
    def predict(self,X,**kwargs):
        return self.model.predict(X)
    
    def get_weights(self,**kwargs):
        
        return self.shap_values.mean(0)