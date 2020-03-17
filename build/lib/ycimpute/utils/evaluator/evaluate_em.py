import copy
import numpy as np

from ycimpute.utils import evaluate
from ycimpute.imputer import EM
from ycimpute.datasets import load_data

class Evaluate(object):
    def __init__(self):
        pass

    def evaluate(self, X_mis,X_full):
        missing_index = evaluate.get_missing_index(np.isnan(X_mis))
        original_arr = X_full[missing_index]
        em_X_filled = EM().complete(copy.copy(X_mis))
        em_filled_arr = em_X_filled[missing_index]
        rmse_em_score = evaluate.RMSE(original_arr, em_filled_arr)
        return rmse_em_score

if __name__ == '__main__':
    boston_mis, boston_full = load_data.load_boston()
    iris_mis, iris_ful = load_data.load_iris()

    boston_score = Evaluate().evaluate(boston_mis, boston_full)
    iris_score = Evaluate().evaluate(iris_mis, iris_ful)
    print(boston_score)
    print(iris_score)