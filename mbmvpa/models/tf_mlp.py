import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2
import numpy as np
from ..data.tf_generator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tempfile
import random
import os
from pathlib import Path

from mbmvpa.models.mvpa_general import MVPA_Base, MVPA_CV
from mbmvpa.utils.report import build_base_report_functions


class MVPACV_MLP(MVPA_CV):
    
    r"""
    
    **MVPACV_MLP** is for providing cross-validation (CV) framework with Multi-layer Perceptron (MLP) as an MVPA model.
    The model is implemented upon Tensorflow (>= 2.0.0).
    Users can choose the option for CV (e.g. 5-fold or leave-one-subject-out), and the model specification.
    Also, users can modulate the configuration for reporting function which includes making brain map (nii), 
    and plots.
    
    Parameters
    ----------
    
    X_dict : dict{str : numpy.ndarray}
        Dictionary for the input voxel feature data which can be indexed by subject IDs.
        Each voxel feature array should be in shape of [time len, voxel feature name]
    y_dict : dict{str : numpy.ndarray}
        Dictionary for the input latent process signals which can be indexed by subject IDs.
        Each signal should be in sahpe of [time len, ]
    voxel_mask : nibabel.nifti1.Nifti1Image
        Brain mask image (nii) used for masking the fMRI images. It will be used to reconstruct a 3D image
        from flattened array of model weights.
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
        Indictator to save results or not
    cv_save_path : str or pathlib.PosixPath, default="."
        Path for saving results
    experiment_name : str, default="unnamed"
        Name for a single run of this analysis
        It will be included in the name of the report folder created.
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
    map_type : str, default="z"
        Type of making brain map. 
            - "z" : z-map will be created using all the weights from CV experiment.
            - "t" : t-map will be created using all the weights from CV experiment.
    sigma : float, default=1
        Sigma value for running Gaussian smoothing on each of reconstructed maps, 
        before integrating maps to z- or t-map.
    
    """
    
    
    def __init__(self,
                 X_dict,
                 y_dict,
                 voxel_mask,
                 method='5-fold',
                 n_cv_repeat=1,
                 cv_save=True,
                 cv_save_path=".",
                 experiment_name="unnamed",
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
                 gpu_visible_devices = None,
                 map_type='z',
                 sigma=1):
    
        input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]

        self.model = MVPA_MLP(input_shape=input_shape,
                             layer_dims=layer_dims,
                             activation=activation,
                             activation_output=activation_output,
                             dropout_rate=dropout_rate,
                             val_ratio=val_ratio,
                             optimizer=optimizer,
                             loss=loss,
                             learning_rate=learning_rate,
                             n_epoch=n_epoch,
                             n_patience=n_patience,
                             n_batch=n_batch,
                             n_sample=n_sample,
                             use_bias=use_bias,
                             gpu_visible_devices=gpu_visible_devices)

        self.report_function_dict = build_base_report_functions(voxel_mask=voxel_mask,
                                                                     experiment_name=experiment_name,
                                                                     map_type=map_type,
                                                                     sigma=sigma)
        super().__init__(X_dict=X_dict,
                        y_dict=y_dict,
                        model=self.model,
                        method=method,
                        n_cv_repeat=n_cv_repeat,
                        cv_save=cv_save,
                        cv_save_path=cv_save_path,
                        experiment_name=experiment_name,
                        report_function_dict=self.report_function_dict)
    
    

class MVPA_MLP(MVPA_Base):
    
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
        self.model = None
        if gpu_visible_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"]=",".join([str(v) for v in gpu_visible_devices])
    
    
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
