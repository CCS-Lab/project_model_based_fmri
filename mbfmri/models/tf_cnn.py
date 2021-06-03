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

from mbfmri.models.mvpa_general import MVPA_Base, MVPA_CV
from mbfmri.utils.report import Reporter

class MVPACV_CNN(MVPA_CV):
    
    r"""
    
    **MVPACV_MLP** is for providing cross-validation (CV) framework with Convolutional Neural Network (CNN) as an MVPA model.
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
                 gpu_visible_devices = None,
                 map_type='z',
                 sigma=1):
    
        input_shape = X_dict[list(X_dict.keys())[0]].shape[1:]

        self.model = MVPA_CNN(input_shape=input_shape,
                             layer_dims=layer_dims,
                             kernel_size=kernel_size,
                             logit_layer_dim=logit_layer_dim,
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
                             batch_norm=batch_norm,
                             voxel_mask=voxel_mask,
                             gpu_visible_devices=gpu_visible_devices)

        self.reporter = Reporter(reports=['brainmap','pearsonr'],
                                 voxel_mask=voxel_mask,
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
                        reporter=self.reporter)
    
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
                 voxel_mask,
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
        self.model = None
        self.voxel_mask = voxel_mask
        if not isinstance(voxel_mask, np.ndarray):
            self.voxel_mask = voxel_mask.get_fdata()
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