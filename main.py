import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer

from models import LR
from utils import read_data


fea_index, fea_value, label = read_data(sys.argv[1])
size = fea_index.shape[0]
train_size = int(0.7 * size)
print(train_size)

Xi_train, Xv_train, y_train = fea_index[:train_size], fea_value[:train_size], label[:train_size]
Xi_valid, Xv_valid, y_valid = fea_index[train_size+1:], fea_value[train_size+1:], label[train_size+1:]

feature_size=500
field_size = Xi_train.shape[1]

params = { 
    "feature_size": feature_size,
    "field_size": field_size,
    "epoch": 50,
    "batch_size": 8,
    "learning_rate": 0.001,
    "optimizer_type": "adam",
    "l2_reg": 0.01,
    "verbose": True
}

lr = LR(**params)
lr.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
