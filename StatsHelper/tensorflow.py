'''
Tensorflow modeling
Version: 1.0
https://www.tensorflow.org/tutorials/keras/basic_regression

This module is used for modeling with Tensorflow
Regression, classification, text classification
'''

# Lib's needed 
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from __future__ import absolute_import, division, print_function

def split_data(df):
    # Split the data up to training and test set's
    train_dataset = df.sample(frac=0.8, random_state=0)
    test_dataset = df.drop(train_dataset.index)
    # Look at the training set
    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    print(train_stats)

    return (train_dataset, test_dataset)

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']
