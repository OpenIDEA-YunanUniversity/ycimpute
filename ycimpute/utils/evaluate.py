
import numpy as np

from sklearn.metrics import accuracy_score

def get_missing_index(mask_all):
    return np.where(mask_all==True)


def accuracy(original, filled):
    score = accuracy_score(original, filled)
    return score

def RMSE(original, filled):
    from sklearn.metrics import mean_squared_error
    score = np.sqrt(mean_squared_error(original, filled))
    return score
