

from __future__ import absolute_import, print_function, division

import numpy as np

from .normalized_distance import all_pairs_normalized_distances


def knn_initialize(
        X,
        missing_mask,
        min_dist=1e-6,
        max_dist_multiplier=1e6):
    """
    Fill X with NaN values if necessary, construct the n_samples x n_samples
    distance matrix and set the self-distance of each row to infinity.

    Returns contents of X laid out in row-major, the distance matrix,
    and an "effective infinity" which is larger than any entry of the
    distance matrix.
    """
    X_row_major = X.copy("C")
    if missing_mask.sum() != np.isnan(X_row_major).sum():
        # if the missing values have already been zero-filled need
        # to put NaN's back in the data matrix for the distances function
        X_row_major[missing_mask] = np.nan
    D = all_pairs_normalized_distances(X_row_major)
    D_finite_flat = D[np.isfinite(D)]
    if len(D_finite_flat) > 0:
        max_dist = max_dist_multiplier * max(1, D_finite_flat.max())
    else:
        max_dist = max_dist_multiplier
    # set diagonal of distance matrix to a large value since we don't want
    # points considering themselves as neighbors
    np.fill_diagonal(D, max_dist)
    D[D < min_dist] = min_dist  # prevents 0s
    D[D > max_dist] = max_dist  # prevents infinities
    return X_row_major, D, max_dist
