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
        root (str or pathlib.Path): data path, if None, must be specified X, y, time_mask_path.
              default path is imported from layout.
        time_mask_path (str or pathlib.Path): optional, time mask data path, if None, default is BIDS_root/derivates/data.

    Returns:
        X (numpy.array): X, which is adjusted dimension and masked time points for training with shape: data # x voxel #
        y (numpy.array): y, which is adjusted dimension and masked time points for training with shape: data # 
    """

    def _load_and_reshape(data_p):
        """
        Load preprocessed fMRI image data and reshape it to 2-dimension array
        
        Arguments:
            data_p (str, or pathlib.Path): path for the data

        Returns:
            reshaped_data (numpy.array): loaded and reshaped data with shape (subject # x run # x time_point #) x voxel #
        """

        data = np.load(data_p)
        reshaped_data = data.reshape(-1, data.shape[-1])
        return reshaped_data

    assert (isinstance(root, str)
        or isinstance(root, Path))
    assert (isinstance(time_mask_path, str)
        or isinstance(time_mask_path, Path))

    # if root is given and path for any of X, y is not given, then use default path.
    root = Path(root)
    data_path = Path(root / config.DEFAULT_SAVE_DIR)

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

    assert X.shape == y.shape

    # use data only at the timepoints indicated in time_mask file.
    if time_masking:
        time_mask = np.load(
            data_path / "time_mask.npy", allow_pickle=True)
        time_mask = np.concatenate(time_mask, 0)
        time_mask = time_mask.flatten()

        X = X[time_mask > 0]
        y = y[time_mask > 0]
        assert X.shape == y.shape

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

    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(X.shape[0])

        self.on_epoch_end()

    # for printing the statistics of the function
    def on_epoch_end(self):
        "Updates indexes after each epoch"

        if self.shuffle:
            np.random.shuffle(self.indexes)

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
