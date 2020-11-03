#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author: Yedarm Seong
@contact: mybirth0407@gmail.com
@last modification: 2020.11.03
"""

import numpy as np
from tensorflow.keras.utils import Sequence


class DataGenerator(Sequence):
    def __init__(self, X, y, batch_size, shuffle=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(X.shape[0])
        self.on_epoch_end()
        
    # for printing the statistics of the function
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    
    def __len__(self):
        "Denotes the number of batches per epoch"
        
        return len(self.indexes) // self.batch_size

    def __getitem__(self, index):  # index : batch no.
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        images = [self.X[i] for i  in indexes]
        targets = [self.y[i] for i in indexes]
        images = np.array(images)
        targets = np.array(targets)

        return images, targets  # return batch