#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

from pathlib import Path

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import Sequence
from ..utils import config


def prepare_dataset(root, time_masking=True, voxel_masking=True):
    """
    Get dataset for fitting model
    """
    """
    Arguments:
        root (str or pathlib.Path): Path where created data (X, y, time_mask, voxel_mask) is stored.
            default path is imported from layout.
        time_masking (bool): Whether to do time masking or not.
        voxel_masking (bool):Whether to do voxel masking or not.

    Returns:
        X (numpy.ndarray): X, which is adjusted dimension and masked time points for training with shape: data # x voxel #.
        y (numpy.ndarray): y, which is adjusted dimension and masked time points for training with shape: data #.
        voxel_mask (nibabel.nifti1.Nifti1Image): Voxel mask file for get result brain map.
    """

    def _load_and_reshape(data_p):
        """
        Load preprocessed fMRI image data and reshape it to 2-dimension array
        
        Arguments:
            data_p (str, or pathlib.Path): path for the data

        Returns:
            reshaped_data (numpy.ndarray): loaded and reshaped data with shape (subject # x run # x time_point #) x voxel #
        """

        data = np.load(data_p)
        reshaped_data = data.reshape(-1, data.shape[-1])
        return reshaped_data

    ###########################################################################
    # parameter check
    assert (isinstance(root, str)
        or isinstance(root, Path))
    assert isinstance(time_masking, bool)
    assert isinstance(voxel_masking, bool)
    ###########################################################################

    data_path = Path(root) / config.DEFAULT_SAVE_DIR

    # aggregate X fragmented by subject to one matrix.
    X_list = list(data_path.glob(f"{config.DEFAULT_FEATURE_PREFIX}_*.npy"))
    X_list.sort(key=lambda x: int(str(x).split('_')[-1].split('.')[0]))

    X = np.concatenate([_load_and_reshape(data_p) for data_p in X_list], 0)
    # makes X to 2-d array with numpy.reshape.
    X = X.reshape(-1, X.shape[-1])

    y = np.load(data_path / "y.npy", allow_pickle=True)
    y = np.concatenate(y, 0)
    # Same as reshape, but use numpy.flatten() to emphasize that y is single value.
    # numpy.flatten makes it 1-d array.
    y = y.flatten()

    assert X.shape[0] == y.shape[0]

    # use data only at the timepoints indicated in time_mask file.
    if time_masking:
        time_mask = np.load(
            data_path / "time_mask.npy", allow_pickle=True)
        time_mask = np.concatenate(time_mask, 0)
        time_mask = time_mask.flatten()

        X = X[time_mask > 0]
        y = y[time_mask > 0]
        assert X.shape[0] == y.shape[0]

    if voxel_masking:
        voxel_mask = nib.load(data_path / config.DEFAULT_VOXEL_MASK_FILENAME)
    else:
        voxel_mask = None

    return X, y, voxel_mask


class DataGenerator(Sequence):
    """
    Data generator required for fitting Keras model. This is just a
    simple wrapper of generating preprocessed fMRI data (:math:`X`) and BOLD-like
    target data (:math:`y`).
    
    Please refer to the below links for examples of using DataGenerator for Keras deep learning framework.
        - https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
        
    Also, this class is used to generate a chunk of data called 'batch', 
    which means a fragment aggregatin the specified number ('batch_size') of data (X,y).
    This partitioning data to small size is intended for utilizing the mini-batch gradient descent (or stochastic gradient descent).
    Please refer to the below link for the framework.
        - https://www.stat.cmu.edu/~ryantibs/convexopt/lectures/stochastic-gd.pdf
    # TODO find a better reference
    """

    def __init__(self, X, y, batch_size, shuffle=True, use_bipolar_balancing=False, **kwargs):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(X.shape[0])
        self.use_bipolar_balancing=use_bipolar_balancing
        if self.use_bipolar_balancing:
            self.ticketer = get_bipolarized_ticketer(y.flatten(),**kwargs)
            self.X_original = X
            self.y_original = y
        self.on_epoch_end()

    # for printing the statistics of the function
    def on_epoch_end(self):
        "Updates indexes after each epoch"

        if self.shuffle:
            np.random.shuffle(self.indexes)
            
        if self.use_bipolar_balancing:
            sample_ids = weighted_sampling(self.y_original, self.ticketer)
            self.X = self.X_original[sample_ids]
            self.y = self.y_original[sample_ids]
            
    def __len__(self):
        "Denotes the number of batches per epoch"
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):
        "Get a batch of data X, y"
        # index : batch no.
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        images = [self.X[i] for i in indexes]
        targets = [self.y[i] for i in indexes]
        images = np.array(images)
        targets = np.array(targets)

        return images, targets  # return batch

def gaussian(x, mean, std):
    return ((2*np.pi*(std**2))**(-.5))*np.exp(-.5*(((x-mean)/std)**2))
    
def get_bipolarized_ticketer(array,high_rate=.1,low_rate=.1, max_val=None, min_val=None, bins=100, max_ticket=10):
    d = array.copy().flatten()
    d.sort()
    low_part = d[:int(len(d)*low_rate)]
    low_part = np.concatenate([low_part,low_part.max()*2 -low_part],0)
    high_part = d[-int(len(d)*high_rate):]
    high_part = np.concatenate([high_part,high_part.min()*2 -high_part],0)
    
    low_mean = low_part.mean()
    low_std = low_part.std()
    
    high_mean = high_part.mean()
    high_std = high_part.std()
    
    if max_val is None:
        max_val = d[-1]
    if min_val is None:
        min_val = d[0]
    
    x = np.linspace(min_val, max_val, bins)
    
    weights = gaussian(x, low_mean, low_std) + gaussian(x, high_mean, high_std)
    weight_max = weights.max()
    ticketer = lambda v: int(((gaussian(v, low_mean, low_std) + \
                              gaussian(v, high_mean, high_std)) /weight_max+1/max_ticket) * max_ticket)
    
    return ticketer

def weighted_sampling(y, ticketer, n_sample=None):
    
    if n_sample is None:
        n_sample = len(y)
    
    pool = []
    
    for i,v in enumerate(y.flatten()):
        pool += [i]*ticketer(v)
    
    sample_ids  = np.random.choice(pool,n_sample)
        
    return sample_ids

def get_binarizing_thresholds(array,high_rate=.1,low_rate=.1, max_val=None, min_val=None, bins=100, max_ticket=10):
    d = array.copy().flatten()
    d.sort()
    low_part = d[:int(len(d)*low_rate)]
    low_part = np.concatenate([low_part,low_part.max()*2 -low_part],0)
    high_part = d[-int(len(d)*high_rate):]
    high_part = np.concatenate([high_part,high_part.min()*2 -high_part],0)
    
    low_mean = low_part.mean()
    low_std = low_part.std()
    
    high_mean = high_part.mean()
    high_std = high_part.std()
    
    if max_val is None:
        max_val = d[-1]
    if min_val is None:
        min_val = d[0]
    
    x = np.linspace(min_val, max_val, bins)
    
    weights = gaussian(x, low_mean, low_std) + gaussian(x, high_mean, high_std)
    weight_max = weights.max()
    ticketer = lambda v: int(((gaussian(v, low_mean, low_std) + \
                              gaussian(v, high_mean, high_std)) /weight_max+1/max_ticket) * max_ticket)
    
    return ticketer