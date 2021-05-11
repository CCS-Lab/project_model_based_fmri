#!/usr/bin/env python3
# -*- coding: utf-8 -*-

## author: Yedarm Seong, Cheoljun cho
## contact: mybirth0407@gmail.com, cjfwndnsl@gmail.com
## last modification: 2020.12.17

import numpy as np
from tensorflow.keras.utils import Sequence

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

    def __init__(self, X, y, batch_size, shuffle=True, use_bipolar_balancing=False, binarize=False,**kwargs):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(X.shape[0])
        self.binarize= binarize
        if self.binarize:
            use_bipolar_balancing = True
            if 'high_rate' in kwargs.key():
                high_rate = kwargs['high_rate']
            else:
                high_rate = None
                
            if 'low_rate' in kwargs.key():
                low_rate = kwargs['low_rate']
            else:
                low_rate = None
                
            self.binarizer = get_binarizer(y.flatten(),high_rate,low_rate)
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
        if self.binarize:
            targets = [self.binarizer(self.y[i]) for i in indexes]
        else:
            targets = [self.y[i] for i in indexes]
        images = np.array(images)
        targets = np.array(targets)

        return images, targets  # return batch
