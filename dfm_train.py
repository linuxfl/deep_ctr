import sys
import numpy as np
from DFM import DFM

token2idx = {}
idx2token = {}

def read_data(filename):
    fea_index = []
    fea_value = []
    label = []
    delay_time = []
    token_size = 0
    with open(filename) as f_in:

        for raw in f_in:
            fi = []
            fv = []
            line = raw.strip("\n\r ").split() 
            label.append(int(line[0]))
            delay_time.append(float(line[1]))
            #dummpy format data
            if len(line[1].split(":")) == 1:
                for v in line[2:]:
                    if int(v) not in token2idx:
                        token2idx[int(v)] = token_size
                        token_size += 1
                        fi.append(token_size - 1)
                    else:
                        fi.append(token2idx[int(v)])
                fv = [ 1 for i in range(len(fi)) ]
            else:
                for kv in line[2:]:
                    kv_sp = kv.split(":")
                    if int(kv_sp[0]) not in token2idx:
                        token2idx[int(kv_sp[0])] = token_size
                        token_size += 1
                        fi.append(token_size - 1)
                    fv.append(int(kv_sp[1]))
            fea_index.append(fi)
            fea_value.append(fv)
    return np.array(fea_index), np.array(fea_value), label, delay_time

idx2token = dict(zip((value, key) for key, value in token2idx.items()))
fea_index, fea_value, label, delay_time = read_data(sys.argv[1])

size = fea_index.shape[0]
train_size = int(0.7 * size)
print(train_size)

Xi_train, Xv_train, y_train, y_tr_dt = fea_index[:train_size], fea_value[:train_size], label[:train_size], delay_time[:train_size]
Xi_valid, Xv_valid, y_valid, y_va_dt = fea_index[train_size+1:], fea_value[train_size+1:], label[train_size+1:], delay_time[train_size+1:]

feature_size=len(token2idx) + 1
field_size = Xi_train.shape[1]


lr_params = { 
    "feature_size": feature_size,
    "field_size": field_size,
    "epoch": 10,
    "batch_size": 8,
    "learning_rate": 0.001,
    "optimizer_type": "gd"
}
dfm = DFM(**lr_params)
dfm.fit(Xi_train, Xv_train, y_train, y_tr_dt, Xi_valid, Xv_valid, y_valid)
