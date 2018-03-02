

from __future__ import absolute_import, print_function, division
import numpy as np

#**********************************************
from ..unsupervised.knn import knn_impute_few_observed, knn_impute_with_argpartition
from ..utils.tools import Solver
from ..utils import config

class KNN(Solver):

    def __init__(
            self,
            k=5,
            orientation="rows",
            use_argpartition=False,
            print_interval=100,
            min_value=None,
            max_value=None,
            normalizer=None,
            verbose=True):
        """
        Parameters
        ----------
        k : int
            Number of neighboring rows to use for imputation.

        orientation : str
            Which axis of the input matrix should be treated as a sample
            (default is "rows" but can also be "columns")

        use_argpartition : bool
           Use a more naive implementation of kNN imputation whichs calls
           numpy.argpartition for each row/column pair. May give NaN if fewer
           than k neighbors are available for a missing value.

        print_interval : int

        min_value : float
            Minimum possible imputed value

        max_value : float
            Maximum possible imputed value

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
        """
        super(KNN, self).__init__(
            self,
            min_value=min_value,
            max_value=max_value,
            normalizer=normalizer)
        self.k = k
        self.verbose = verbose
        self.orientation = orientation
        self.print_interval = print_interval
        if use_argpartition:
            self._impute_fn = knn_impute_with_argpartition
        else:
            self._impute_fn = knn_impute_few_observed

    def solve(self, X):
        if self.orientation == "columns":
            missing_mask = self.masker(X)[config.all]

        elif self.orientation != "rows":
            raise ValueError(
                "Orientation must be either 'rows' or 'columns', got: %s" % (
                    self.orientation,))

        X_imputed = self._impute_fn(
            X=X,
            missing_mask=self.masker(X)[config.all],
            k=self.k,
            verbose=self.verbose,
            print_interval=self.print_interval)

        failed_to_impute = np.isnan(X_imputed)
        n_missing_after_imputation = failed_to_impute.sum()
        if n_missing_after_imputation != 0:
            if self.verbose:
                print("[KNN] Warning: %d/%d still missing after imputation, replacing with 0" % (
                    n_missing_after_imputation,
                    X.shape[0] * X.shape[1]))
            X_imputed[failed_to_impute] = X[failed_to_impute]

        if self.orientation == "columns":
            X_imputed = X_imputed.T


        return X_imputed
