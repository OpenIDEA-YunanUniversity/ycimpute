

from __future__ import absolute_import, print_function, division

from six.moves import range
import numpy as np

def all_pairs_normalized_distances(X):
    """
    We can't really compute distances over incomplete data since
    rows are missing different numbers of entries.
    The next best thing is the mean squared difference between two vectors
    (a normalized distance), which gets computed only over the columns that
    two vectors have in common. If two vectors have no features in common
    then their distance is infinity.

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n_samples, n_features) with missing entries
        marked using np.nan

    Returns a (n_samples, n_samples) matrix of pairwise normalized distances.
    """
    n_rows, n_cols = X.shape

    # matrix of mean squared difference between between samples
    D = np.ones((n_rows, n_rows), dtype="float32", order="C") * np.inf

    # we can cheaply determine the number of columns that two rows share
    # by taking the dot product between their finite masks
    observed_elements = np.isfinite(X).astype(int)
    n_shared_features_for_pairs_of_rows = np.dot(
        observed_elements,
        observed_elements.T)
    no_overlapping_features_rows = n_shared_features_for_pairs_of_rows == 0
    number_incomparable_rows = no_overlapping_features_rows.sum(axis=1)
    row_overlaps_every_other_row = (number_incomparable_rows == 0)
    row_overlaps_no_other_rows = number_incomparable_rows == n_rows
    valid_rows_mask = ~row_overlaps_no_other_rows
    valid_row_indices = np.where(valid_rows_mask)[0]

    # preallocate all the arrays that we would otherwise create in the
    # following loop and pass them as "out" parameters to NumPy ufuncs
    diffs = np.zeros_like(X)
    missing_differences = np.zeros_like(diffs, dtype=bool)
    valid_rows = np.zeros(n_rows, dtype=bool)
    ssd = np.zeros(n_rows, dtype=X.dtype)

    for i in valid_row_indices:
        x = X[i, :]
        np.subtract(X, x.reshape((1, n_cols)), out=diffs)
        np.isnan(diffs, out=missing_differences)

        # zero out all NaN's
        diffs[missing_differences] = 0

        # square each difference
        diffs **= 2

        observed_counts_per_row = n_shared_features_for_pairs_of_rows[i]

        if row_overlaps_every_other_row[i]:
            # add up all the non-missing squared differences
            diffs.sum(axis=1, out=D[i, :])
            D[i, :] /= observed_counts_per_row
        else:
            np.logical_not(no_overlapping_features_rows[i], out=valid_rows)

            # add up all the non-missing squared differences
            diffs.sum(axis=1, out=ssd)
            ssd[valid_rows] /= observed_counts_per_row[valid_rows]
            D[i, valid_rows] = ssd[valid_rows]
    return D


def all_pairs_normalized_distances_reference(X):
    """
    Reference implementation of normalized all-pairs distance, used
    for testing the more efficient implementation above for equivalence.
    """
    n_samples, n_cols = X.shape
    # matrix of mean squared difference between between samples
    D = np.ones((n_samples, n_samples), dtype="float32") * np.inf
    for i in range(n_samples):
        diffs = X - X[i, :].reshape((1, n_cols))
        missing_diffs = np.isnan(diffs)
        missing_counts_per_row = missing_diffs.sum(axis=1)
        valid_rows = missing_counts_per_row < n_cols
        D[i, valid_rows] = np.nanmean(
            diffs[valid_rows, :] ** 2,
            axis=1)
    return D
