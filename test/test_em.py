import pandas as pd
import numpy as np

from  ycimpute.imputer import EM
from .metrics import RMSE
from .generate_data import missing_data,missing_mask,complete_data
from ycimpute.utils.normalizer import min_max_scale

def test_em():
    X_filled = EM().complete(missing_data)
    complete_data_, _, _ = min_max_scale(complete_data)
    X_filled, _, _ = min_max_scale(X_filled)

    score = RMSE(complete_data_[missing_mask],
                 X_filled[missing_mask])
    print(score)
