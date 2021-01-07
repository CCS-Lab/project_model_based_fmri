#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#author: Cheol Jun Cho, Yedarm Seong
#contact: cjfwndnsl@gmail.com, mybirth0407@gmail.com
#last modification: 2020.11.03

import datetime
from pathlib import Path
import numpy as np

from bids import BIDSLayout
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2

from ..data import loader
from ..data.loader import prepare_dataset
from ..utils import config

import pdb

    
class DefaultExtractor():
    # model is assumed to be linear
    
    def __init__(self, input_shape, n_sample=1):
        self.input_shape = input_shape
        self.n_sample = 1
        
    def __call__(self,model,batch_size=256):
        
        outputs_pool = []
        for i in range(self.n_sample):
        
            sample = np.eye(self.input_shape)
            n_step = int(np.ceil((self.input_shape+0.0)/batch_size))

            outputs = []
            for i in range(n_step):
                output = model.predict(sample[i*batch_size:(i+1)*batch_size])
                output = list(output.flatten())
                outputs += output

            outputs = np.array(outputs)[:self.input_shape]
            outputs_pool.append(outputs)
        
        mean_outputs = np.array(outputs_pool).mean(0)
        
        return mean_outputs
        
    
    
class Regressor_TF():
    ''' Model for MVPA regression
    
    Model abstraction for  MVPA regression model. 
    The implemented model should have 
    
    Attributes:
        X (numpy.ndarray): preprocessed fMRI data. shape : data # x voxel #
        y (numpy.ndarray): parametric modulation values to regress X against. shape: data #
        model (tf.model) : regression model for MVPA
        layout (bids.BIDSLayout): BIDSLayout by bids package. used to get default save_path.
        save_path (Path): root path for saving fitting info. default would be current working dir. 
        plot_path (Path): path for saving plot. it will be *save_path/plot*
        log_path (Path): path for saving fitting log. it wiil be *save_path/log*
        save (bool): if true save all intermittent results.# TODO. not used for now.  
        verbose (int): if > 0 then log fitting process and report a validation mse of each repetition. #TODO add more options
        n_repeat (int): the number of repetition for training models. you will get *n_repeat* number of coefficients.
        n_sample (int): maximum number of instance of data (X,y) used in a single repetition. 
        
    Methods:
        run (callable): method for fitting model with data (X,y) and return coefficients
        
    '''
    
    def __init__(self,
                 X=None,
                 y=None,
                 voxel_mask=None,
                 model=None,
                 model_name=None,
                 extractor=None,
                 extract_n_sample=1,
                 root=None,
                 layout=None,
                 save_path=None,
                 save=True,
                 verbose=1,
                 n_repeat=5,
                 n_sample=10000,
                 n_epoch=100,
                 n_patience=10,
                 n_batch=64,
                 validation_split_ratio=0.2
                 ):
        
        if root is not None:
            layout = BIDSLayout(root, derivatives=True)
            
        if save_path is None:
            if layout is None:
                self.save_path = Path('.')
            else:
                sp = Path(
                    layout.derivatives["fMRIPrep"].root)\
                    / config.DEFAULT_SAVE_PATH_CKPT 
                if not sp.exists():
                    sp.mkdir()
                sp = sp / model_name
                if not sp.exists():
                    sp.mkdir()
                self.save_path = sp
        else:
            self.save_path = Path(save_path)
            
        if ( X is None or y is None ) and layout is not None:
            X, y, voxel_mask = prepare_dataset(layout.derivatives["fMRIPrep"].root)
        
        self.layout=layout
        self.X = X
        self.y = y
        self.voxel_mask = voxel_mask
        self.model = model
        if extractor is None:
            self.extractor = DefaultExtractor(X.shape[-1],extract_n_sample)
        else:
            self.extractor = extractor
            
        self.chk_path = None
        self.log_path = None
        self.save = save
        self.verbose = verbose
        self.n_repeat = n_repeat
        self.n_sample = n_sample
        self.n_epoch = n_epoch
        self.n_patience = n_patience
        self.n_batch = n_batch
        self.validation_split_ratio = validation_split_ratio
        self._coeffs = []
        self._errors = []
        self._make_log_dir()
        
    def _make_log_dir(self):
        now = datetime.datetime.now()
        save_root = self.save_path / f'report_{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}-{now.second}'
        self.chk_path = save_root / 'chekpoint'
        self.log_path = save_root / 'log'
        
        save_root.mkdir()
        self.chk_path.mkdir()
        self.log_path.mkdir()
        
        return
    
    def _reset_model(self):
        model = self.model
        for layer in model.layers:
            if hasattr(layer, 'init'):
                init = getattr(layer, 'init')
                new_weights = init(layer.get_weights()[0].shape).get_value()
                bias = shared_zeros(layer.get_weights()[1].shape).get_value()
                layer.set_weights([new_weights, bias])
        model.reset_states()
        
        return model
    
    def run(self):
        
        self._coeffs = []
        self._errors = []
        
        for i in range(1, self.n_repeat + 1):
            # random sampling "n_samples" if the given number of X,y instances is bigger
            # than maximum allowed number for training
            np.random.seed(i)
            tf.random.set_seed(i) # also need to set random seed in tensorflow
            ids = np.arange(self.X.shape[0])

            if self.X.shape[0] > self.n_sample:
                np.random.shuffle(ids)
                ids = ids[:self.n_sample]

            # split data to training set and validation set
            train_ids, test_ids = train_test_split(
                ids, test_size=self.validation_split_ratio, random_state=i
            )
            train_steps = len(train_ids) // self.n_batch
            val_steps = len(test_ids) // self.n_batch

            assert train_steps > 0
            assert val_steps > 0
            
            X_train = self.X[train_ids]
            X_test = self.X[test_ids]
            y_train = self.y[train_ids]
            y_test = self.y[test_ids]

            # create helper class for generating data
            # support mini-batch training implemented in Keras
            train_generator = loader.DataGenerator(
                X_train, y_train, self.n_batch, shuffle=True)
            val_generator = loader.DataGenerator(
                X_test, y_test, self.n_batch, shuffle=False)
            # should be implemented in the actual model
            
            best_model_filepath = str(self.chk_path / \
            f"repeat_{i:0{len(str(self.n_repeat))}}.ckpt")
        
            # temporal buffer for intermediate training results (weights) of training.
            mc = ModelCheckpoint(
                best_model_filepath,
                save_best_only=True, save_weights_only=True,
                monitor="val_loss", mode="min")

            # device for early stopping. if val_loss does not decrease within patience, 
            # the training will stop
            es = EarlyStopping(monitor="val_loss", patience=self.n_patience)
            
            model = self._reset_model()
            
            model.fit(train_generator, epochs=self.n_epoch,
                  verbose=0, callbacks=[mc, es],
                  validation_data=val_generator,
                  steps_per_epoch=train_steps,
                  validation_steps=val_steps)
            
            # load best model
            model.load_weights(best_model_filepath)
            # validation 
            y_pred = model.predict(X_test)
            error = mean_squared_error(y_pred, y_test)
            if self.verbose > 0:
                print(f"[{i}/{self.n_repeat}] - val_loss: {error:.04f}")
            self._errors.append(error)
            
            # extracting voxel-wise mapped weight (coefficient) map
            coeff = self.extractor(model)
            self._coeffs.append(coeff)
    

        self._coeffs = np.array(self._coeffs)
        self._errors = np.array(self._errors)
        
        return self._coeffs
    
    
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
                    input_shape=(input_shape,),
                    use_bias=False))
    model.add(Dropout(dropout_rate))

    # add layers
    for dim in layer_dims[1:]:
        model.add(Dense(dim, activation=activation, use_bias=False))
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

class MLP(Regressor_TF):
    
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
        input_shape = self.X.shape[-1] 
        self.model = build_mlp(input_shape=input_shape,
                             layer_dims=layer_dims,
                             activation=activation,
                             activation_output=activation_output,
                             dropout_rate=dropout_rate,
                             optimizer=optimizer,
                             loss=loss)
        
        if not use_default_extractor:
            self.extractor = extractor_mlp
        