

from __future__ import absolute_import, print_function, division
import time

import numpy as np

from six.moves import range

from .common import knn_initialize

def knn_impute_with_argpartition(
        X,
        missing_mask,
        k,
        verbose=False,
        print_interval=100):
    """
    Fill in the given incomplete matrix using k-nearest neighbor imputation.

    This version is a simpler algorithm meant primarily for testing but
    surprisingly it's faster for many (but not all) dataset sizes, particularly
    when most of the columns are missing in any given row. The crucial
    bottleneck is the call to numpy.argpartition for every missing element
    in the array.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool

    Returns a row-major copy of X with imputed values.
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    # put the missing mask in column major order since it's accessed
    # one column at a time
    missing_mask_column_major = np.asarray(missing_mask, order="F")
    X_row_major, D, effective_infinity = \
        knn_initialize(X, missing_mask)
    D_reciprocal = 1.0 / D

    dot = np.dot
    array = np.array
    argpartition = np.argpartition

    for i in range(n_rows):
        missing_indices = np.where(missing_mask[i])[0]

        if verbose and i % print_interval == 0:
            print(
                "Imputing row %d/%d with %d missing, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_indices),
                    time.time() - start_t))
        d = D[i, :]
        inv_d = D_reciprocal[i, :]
        for j in missing_indices:
            # move rows which lack this feature to be infinitely far away
            d_copy = d.copy()
            d_copy[missing_mask_column_major[:, j]] = effective_infinity

            neighbor_indices = argpartition(d_copy, k)[:k]
            if d_copy[neighbor_indices].max() >= effective_infinity:
                # if there aren't k rows with the feature of interest then
                # we need to filter out indices of points at infinite distance
                neighbor_indices = array([
                    neighbor_index
                    for neighbor_index in neighbor_indices
                    if d_copy[neighbor_index] < effective_infinity
                ])
            n_current_neighbors = len(neighbor_indices)

            if n_current_neighbors > 0:
                neighbor_weights = inv_d[neighbor_indices]
                X_row_major[i, j] = (
                    dot(X[:, j][neighbor_indices], neighbor_weights) /
                    neighbor_weights.sum()
                )
    return X_row_major
