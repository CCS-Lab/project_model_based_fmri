#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@main author: Cheol Jun Cho
@code optimization: Yedarm Seong
@contact: cjfwndnsl@gmail.com
          mybirth0407@gmail.com
@last modification: 2020.11.03
"""

import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow import keras as K
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
from tensorflow.keras.layers import Dense, BatchNormalization, ReLU, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import  ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_1samp
from ..data import DataGenerator

from sklearn.metrics import mean_squared_error

import logging


logging.basicConfig(level=logging.INFO)

