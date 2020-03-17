
import pandas as pd
import numpy as np
import copy
import h5py
##################################################

from ...imputer.mice import MICE
from ...imputer.knnimput import KNN
from ...imputer.iterforest import IterImput
from ...imputer.simple import SimpleFill
from ...imputer import EM

from ...utils.tools import Solver
from .. import evaluate
from .. import config

solver = Solver()

def analysiser(missing_X, original_X):
    missing_X = np.asarray(missing_X)
    original_X = np.asarray(original_X)

    mask_all = solver.masker(missing_X)[config.all]
    missing_index = evaluate.get_missing_index(mask_all)
    original_arr = original_X[missing_index]

    ##################################################

    mice_X_filled = MICE().complete(copy.copy(missing_X))
    mice_filled_arr = mice_X_filled[missing_index]
    rmse_mice_score = evaluate.RMSE(original_arr, mice_filled_arr)

    #########################################################
    iterforest_X_filled = IterImput().complete(copy.copy(missing_X))
    iterforest_filled_arr = iterforest_X_filled[missing_index]
    rmse_iterforest_score = evaluate.RMSE(original_arr, iterforest_filled_arr)


    ############################################################
    knn_X_filled = KNN(k=3).complete(copy.copy(missing_X))
    knn_filled_arr = knn_X_filled[missing_index]
    rmse_knn_score = evaluate.RMSE(original_arr, knn_filled_arr)

    ######################################################
    mean_X_filled = SimpleFill(fill_method='mean').complete(copy.copy(missing_X))
    mean_filled_arr = mean_X_filled[missing_index]
    rmse_mean_score = evaluate.RMSE(original_arr, mean_filled_arr)
    #################################################################
    zero_X_filled = SimpleFill(fill_method='zero').complete(copy.copy(missing_X))
    zero_filled_arr = zero_X_filled[missing_index]
    rmse_zero_score = evaluate.RMSE(original_arr, zero_filled_arr)

    ################################################
    median_X_filled = SimpleFill(fill_method='median').complete(copy.copy(missing_X))
    median_filled_arr = median_X_filled[missing_index]
    rmse_median_score = evaluate.RMSE(original_arr, median_filled_arr)
    ##########################################################################
    min_X_filled = SimpleFill(fill_method='min').complete(copy.copy(missing_X))
    min_filled_arr = min_X_filled[missing_index]
    rmse_min_score = evaluate.RMSE(original_arr, min_filled_arr)

    #######################################################
    em_X_filled = EM().complete(copy.copy(missing_X))
    em_filled_arr = em_X_filled[missing_index]
    rmse_em_score = evaluate.RMSE(original_arr,em_filled_arr)
    ################################################

    return {'rmse_mice_score':rmse_mice_score,
            'rmse_iterforest_score':rmse_iterforest_score,
            'rmse_knn_score':rmse_knn_score,
            'rmse_mean_score':rmse_mean_score,
            'rmse_zero_score':rmse_zero_score,
            'rmse_median_score':rmse_median_score,
            'rmse_min_score':rmse_min_score,
            'rmse_em_score': rmse_em_score
            }


def example():
    from ...datasets import load_data
    boston_mis, boston_full = load_data.load_boston()
    print(analysiser(boston_mis, boston_full))