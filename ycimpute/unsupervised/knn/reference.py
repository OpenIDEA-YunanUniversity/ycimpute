
from __future__ import absolute_import, print_function, division

import numpy as np
from six.moves import range

from .common import knn_initialize

def knn_impute_reference(
        X,
        missing_mask,
        k,
        verbose=False,
        print_interval=100):
    """
    Reference implementation of kNN imputation logic.
    """
    n_rows, n_cols = X.shape
    X_result, D, effective_infinity = \
        knn_initialize(X, missing_mask)

    for i in range(n_rows):
        for j in np.where(missing_mask[i, :])[0]:
            distances = D[i, :].copy()

            # any rows that don't have the value we're currently trying
            # to impute are set to infinite distances
            distances[missing_mask[:, j]] = effective_infinity
            neighbor_indices = np.argsort(distances)
            neighbor_distances = distances[neighbor_indices]

            # get rid of any infinite distance neighbors in the top k
            valid_distances = neighbor_distances < effective_infinity
            neighbor_distances = neighbor_distances[valid_distances][:k]
            neighbor_indices = neighbor_indices[valid_distances][:k]

            weights = 1.0 / neighbor_distances
            weight_sum = weights.sum()

            if weight_sum > 0:
                column = X[:, j]
                values = column[neighbor_indices]
                X_result[i, j] = np.dot(values, weights) / weight_sum
    return X_result
