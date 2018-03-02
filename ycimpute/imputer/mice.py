
from time import time
import numpy as np
from sklearn.linear_model import LinearRegression

from  ..utils.tools import Solver
from ..utils import config

class MICE(Solver):
    """
        Basic implementation of MICE package from R.
        This version assumes all of the columns are ordinal,
        and uses ridge regression.

            Parameters
            ----------
            visit_sequence : str
                Possible values: "monotone" (default), "roman", "arabic",
                    "revmonotone".

            n_imputations : int
                Defaults to 100

            n_burn_in : int
                Defaults to 10

            impute_type : str
                "pmm" (default) is probablistic moment matching.
                "col" means fill in with samples from posterior predictive
                    distribution.

            n_pmm_neighbors : int
                Number of nearest neighbors for PMM, defaults to 5.

            model : predictor function
                A model that has fit, predict, and predict_dist methods.
                Defaults to LinerRegression() from scikit-learn
                Note that the regularization parameter lambda_reg
                is by default scaled by np.linalg.norm(np.dot(X.T,X)).
                Sensible lambda_regs to try: 0.25, 0.1, 0.01, 0.001, 0.0001.

            n_nearest_columns : int
                Number of other columns to use to estimate current column.
                Useful when number of columns is huge.
                Default is to use all columns.

            init_fill_method : str
                Valid values: {"mean", "median", or "random"}
                (the latter meaning fill with random samples from the observed
                values of a column)

            min_value : float
                Minimum possible imputed value

            max_value : float
                Maximum possible imputed value

            verbose : boolean
        """
    def __init__(self,
                 visit_sequence='monotone',
                 n_imputations=100,
                 n_burn_in=10,
                 n_pmm_neighbors=5,
                 impute_type='pmm',
                 model=LinearRegression(),
                 n_nearest_columns=np.infty,
                 init_fill_method="mean",
                 min_value=None,
                 max_value=None,
                 verbose=False):


        super(MICE, self).__init__(n_imputations=n_imputations,
                             min_value=min_value,
                             max_value=max_value)

        self.mask_memo_dict = None
        self.visit_sequence = visit_sequence
        self.n_burn_in = n_burn_in
        self.n_pmm_neighbors = n_pmm_neighbors
        self.impute_type = impute_type
        self.model = model
        self.n_nearest_columns = n_nearest_columns
        self.verbose = verbose
        self.fill_method = init_fill_method

    def _imputation_round(self, X_filled, visit_indices):
        global imputed_values
        for col in visit_indices:
            x_obs, y_obs, x_mis = self.split(X_filled, col, self.mask_memo_dict)
            model = self.model
            model.fit(x_obs, y_obs)

            if self.impute_type == 'pmm':
                col_preds_missing = model.predict(x_mis)
                col_preds_observed = model.predict(x_obs)
                D = np.abs(col_preds_missing[:, np.newaxis] - col_preds_observed)
                k = np.minimum(self.n_pmm_neighbors, len(col_preds_observed) - 1)
                k_nearest_indices = np.argpartition(D, k, 1)[:, :k]
                imputed_indices = np.array([
                    np.random.choice(neighbor_index)
                    for neighbor_index in k_nearest_indices])
                imputed_values = y_obs[imputed_indices]
            elif self.impute_type == 'col':
                pass

            X_filled[self.mask_memo_dict[col], col] = imputed_values
        return X_filled

    def clip(self, X, *kwargs):
        """
        Clip values to fall within any global or column-wise min/max constraints
        :param **kwargs:
        """
        if self.min_value is not None:
            X[X < self.min_value] = self.min_value
        if self.max_value is not None:
            X[X > self.max_value] = self.max_value
        return X

    def get_visit_indices(self, missing_mask):
        """
            Decide what order we will update the columns.e.g. sort columns
            As a homage to the MICE package, we will have 4 options of
            how to order the updates.
        """
        n_rows, n_cols = missing_mask.shape
        if self.visit_sequence == 'roman':
            return np.arange(n_cols)
        elif self.visit_sequence == 'arabic':
            return np.arange(n_cols - 1, -1, -1)  # same as np.arange(d)[::-1]
        elif self.visit_sequence == 'monotone':
            return np.argsort(missing_mask.sum(0))[::-1]
        elif self.visit_sequence == 'revmonotone':
            return np.argsort(missing_mask.sum(0))
        else:
            raise ValueError("Invalid choice for visit order: %s" % self.visit_sequence)

    def initialize(self, X):
        self.mask_memo_dict = self.masker(X)
        missing_mask = self.mask_memo_dict[config.all]
        X_filled = self.fill(X, missing_mask, fill_method=self.fill_method)
        return X_filled

    def _multiple_imputations(self, X):
        start_t = time()

        X_filled = self.initialize(X)
        visit_idx = self.sort_col(X)
        missing_mask = self.mask_memo_dict[config.all]
        total_rounds = self.n_burn_in + self.n_imputations

        results_list = []

        for m in range(total_rounds):
            if self.verbose:
                print(
                    "[MICE] Starting imputation round %d/%d, elapsed time %0.3f" % (
                        m + 1,
                        total_rounds,
                        time() - start_t))
            X_filled = self._imputation_round(X_filled, visit_idx)

            if m >= self.n_burn_in:
                results_list.append(X_filled[missing_mask])
        return np.array(results_list), missing_mask


    def solve(self, X):
        if self.verbose:
            print("[MICE] Completing matrix with shape %s" % (X.shape,))
        X_completed = np.array(X.copy())
        imputed_arrays, missing_mask = self._multiple_imputations(X)
        # average the imputed values for each feature
        average_imputated_values = imputed_arrays.mean(axis=0)
        X_completed[missing_mask] = average_imputated_values
        return X_completed

