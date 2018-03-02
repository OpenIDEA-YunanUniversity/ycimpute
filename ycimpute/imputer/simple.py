
from ..utils.tools import Solver
from ..utils import config



class SimpleFill(Solver):
    def __init__(self, fill_method="mean", min_value=None, max_value=None):
        """
        Possible values for fill_method:
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian noise according to mean/std of column
        """
        Solver.__init__(
            self,
            fill_method=fill_method,
            min_value=None,
            max_value=None)

    def solve(self, X):
        """
        Since X is given to us already filled, just return it.
        """
        mask_all = self.masker(X)[config.all]
        X = self.fill(X=X,missing_mask=mask_all)
        return X



