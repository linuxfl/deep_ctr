import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import make_scorer

from models import LR, FM, DeepFM, PNN
from utils import read_data


fea_index, fea_value, label = read_data(sys.argv[1])
size = fea_index.shape[0]
train_size = int(0.7 * size)
print(train_size)

Xi_train, Xv_train, y_train = fea_index[:train_size], fea_value[:train_size], label[:train_size]
Xi_valid, Xv_valid, y_valid = fea_index[train_size+1:], fea_value[train_size+1:], label[train_size+1:]

feature_size=991832
field_size = Xi_train.shape[1]

algo = sys.argv[2]

if algo == 'lr':
    lr_params = { 
        "feature_size": feature_size,
        "field_size": field_size,
        "epoch": 10,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "l2_reg": 0.01,
        "verbose": True
    }
    lr = LR(**lr_params)
    lr.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
elif algo == 'fm':
    fm_params = { 
        "feature_size": feature_size,
        "field_size": field_size,
        "embedding_size": 15,
        "epoch": 20,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "l2_w_reg": 0.01,
        "l2_v_reg": 0.01,
        "verbose": True
    }
    fm = FM(**fm_params)
    fm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
elif algo == 'deepfm':
    deepfm_params = { 
        "feature_size": feature_size,
        "field_size": field_size,
        "embedding_size": 15,
        "deep_layers": [256, 128, 64],
        "epoch": 20,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "l2_reg": 0.01,
        "dropout_deep": [0.5, 0.5, 0.5, 0.5],
        "verbose": True
    }
    deepfm = DeepFM(**deepfm_params)
    deepfm.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)
elif algo == 'pnn':
    pnn_params = { 
        "feature_size": feature_size,
        "field_size": field_size,
        "embedding_size": 15,
        "deep_layers": [256, 128, 64],
        "epoch": 20,
        "batch_size": 1024,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "l2_reg": 0.01,
        "dropout_deep": [0.5, 0.5, 0.5, 0.5],
        "verbose": True
    }
    pnn = PNN(**pnn_params)
    pnn.fit(Xi_train, Xv_train, y_train, Xi_valid, Xv_valid, y_valid)

