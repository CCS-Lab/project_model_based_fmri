# -*- coding: utf-8 -*-
"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.03
"""


from pathlib import Path

import numpy as np
from tensorflow.keras.utils import Sequence


def prepare_dataset(root=None, X_path=None, y_path=None, time_mask_path=None):
    """
    Get dataset for fitting model

    Arguments:
        root: data path, if None, must be specified X, y, time_mask_path.
              default path is imported from layout.
        X_path: optional, X data path, if None, default is bids/derivates/data.
        y_path: optional, y data path, if None, default is bids/derivates/data.
        time_mask_path: optional, time mask data path, if None, default is bids/derivates/data.

    Return:
        TODO: explain to time points
        X: X, which is adjusted dimension and masked time points for training
        y: y, which is adjusted dimension and masked time points for training
    """

    # if root is given and path for any of X, y is not given, then use default path.
    if root is not None:
        root = Path(root)
        X_path = root / 'mvpa'
        y_path = root / 'mvpa'
        time_mask_path = root / 'mvpa'
    else:
        assert X_path is not None or y_path is not None, (
            "If root is None, you must be indicate data path (X, Y, time mask)"
        )

        X_path = Path(X_path)
        y_path = Path(y_path)

    # aggregate X fragmented by subject to one matrix
    X_list = list(X_path.glob('X_*.npy'))
    X_list.sort(key=lambda v: int(str(v).split('_')[-1].split('.')[0]))
    X = np.concatenate([np.load(data_path) for data_path in X_list], 0)
    X = X.reshape(-1, X.shape[-1])

    y = np.load(y_path / 'y.npy', allow_pickle=True)
    y = np.concatenate(y, 0)
    X = X.reshape(-1, X.shape[-1])
    y = y.flatten()

    if time_mask_path is not None:
        # use data only at timepoints indicated in time_mask file.
        time_mask_path = Path(time_mask_path)
        time_mask = np.load(
            time_mask_path / 'time_mask.npy', allow_pickle=True)
        time_mask = np.concatenate(time_mask, 0)
        time_mask = time_mask.flatten()

    X = X[time_mask > 0]
    y = y[time_mask > 0]

    return X, y


class DataGenerator(Sequence):
    """
    Data generator class required for fitting Keras model. This is just a
    simple wrapper of feeding preprocessed fMRI data (:math:`X`) and BOLD-like
    target data (:math:`y`).
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
        # index : batch no.
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]
        images = [self.X[i] for i in indexes]
        targets = [self.y[i] for i in indexes]
        images = np.array(images)
        targets = np.array(targets)

        return images, targets  # return batch
