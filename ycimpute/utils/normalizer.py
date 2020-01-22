

import numpy as np

def min_max_scale(x):
    cols = x.shape[1]
    min_record = []
    max_record = []

    for col in range(cols):
        min_val= np.min(x[:,col])
        max_val = np.max(x[:,col])
        x[:,col] = (x[:,col] - min_val)/(max_val - min_val)
        min_record.append(min_val)
        max_record.append(max_val)

    return x, min_record,max_record

def zero_score_scale(x):
    cols = x.shape[1]
    for col in range(cols):
        x[:,col] = (x[:,col]-np.mean(x[:,col]))/(np.std(x[:,col]))

    return x

def min_max_recover(X, min_vec, max_vec):
    cols = X.shape[1]
    for col in range(cols):
        X[:,col] = X[:,col]*(max_vec[col]-min_vec[col])+min_vec[col]
    return X


NORMALIZERS = {'min_max':min_max_scale,
                'zero_score':zero_score_scale}

RECOVER = {'min_max':min_max_recover}