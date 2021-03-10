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

def get_binarizer(array,high_rate=.1,low_rate=.1):
    d = array.copy().flatten()
    d.sort()
    low_pole = d[int(len(d)*low_rate)]
    high_pole = d[-int(len(d)*high_rate)]
        
    binarizer = lambda v: int((high_mean-v)<(v-low_mean))
    
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