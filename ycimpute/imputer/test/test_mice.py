import pandas as pd
import numpy as np
from datas import dpath

from imputer.mice import MICE
from utils.tools import Solver
from utils import config
from utils import evaluate

sovler = Solver()


def test_get_visit_indices(data):
    mice = MICE()
    mask = sovler.masker(data)[config.all]
    idx = mice.get_visit_indices(mask)
    print(idx)


def test_imputation_round(data):
    mice = MICE()
    visit_idx = sovler.sort_col(data)
    mask = sovler.masker(X=data)[config.all]
    X_filled = mice.initialize(X=data)
    X_filled = mice._imputation_round(X_filled=X_filled, visit_indices=visit_idx)
    print(X_filled)


def test_mice(X_missing, X_original, metrics_method):
    mask_all = sovler.masker(X_missing)[config.all]
    X_filled = MICE().complete(X_missing)
    missing_index = evaluate.get_missing_index(mask_all)
    filled_arr = X_filled[missing_index]
    original_arr = X_original[missing_index]
    score = metrics_method(original_arr, filled_arr)

    return score


if __name__ == "__main__":
    ########BOSTON#################################
    missing_X = pd.read_csv(dpath.boston_missing)
    original_X = pd.read_csv(dpath.boston_full)
    missing_X = np.asarray(missing_X)
    original_X = np.asarray(original_X)

    test_mice(missing_X, original_X, evaluate.RMSE)

