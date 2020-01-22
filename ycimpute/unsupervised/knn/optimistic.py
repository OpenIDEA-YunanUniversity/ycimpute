

from __future__ import absolute_import, print_function, division
import time

from six.moves import range
import numpy as np

from .common import knn_initialize

def knn_impute_optimistic(
        X,
        missing_mask,
        k,
        verbose=False,
        print_interval=100):
    """
    Fill in the given incomplete matrix using k-nearest neighbor imputation.

    This version assumes that most of the time the same neighbors will be
    used so first performs the weighted average of a row's k-nearest neighbors
    and checks afterward whether it was valid (due to possible missing values).

    Has been observed to be a lot faster for 1/4 missing images matrix
    with 1000 rows and ~9000 columns.

    Parameters
    ----------
    X : np.ndarray
        Matrix to fill of shape (n_samples, n_features)

    missing_mask : np.ndarray
        Boolean array of same shape as X

    k : int

    verbose : bool

    Modifies X by replacing its missing values with weighted averages of
    similar rows. Returns the modified X.
    """
    start_t = time.time()
    n_rows, n_cols = X.shape
    X_row_major, D, _ = knn_initialize(X, missing_mask)
    D_sorted_indices = np.argsort(D, axis=1)
    X_column_major = X_row_major.copy(order="F")

    dot = np.dot

    # preallocate array to prevent repeated creation in the following loops
    neighbor_weights = np.ones(k, dtype=X.dtype)

    missing_mask_column_major = np.asarray(missing_mask, order="F")
    observed_mask_column_major = ~missing_mask_column_major

    for i in range(n_rows):
        missing_columns = np.where(missing_mask[i])[0]
        if verbose and i % print_interval == 0:
            print(
                "Imputing row %d/%d with %d missing, elapsed time: %0.3f" % (
                    i + 1,
                    n_rows,
                    len(missing_columns),
                    time.time() - start_t))
        n_missing_columns = len(missing_columns)
        if n_missing_columns == 0:
            continue

        row_distances = D[i, :]
        neighbor_indices = D_sorted_indices[i, :]
        X_missing_columns = X_column_major[:, missing_columns]

        # precompute these for the fast path where the k nearest neighbors
        # are not missing the feature value we're currently trying to impute
        k_nearest_indices = neighbor_indices[:k]
        np.divide(1.0, row_distances[k_nearest_indices], out=neighbor_weights)
        # optimistically impute all the columns from the k nearest neighbors
        # we'll have to back-track for some of the columns for which
        # one of the neighbors did not have a value
        X_knn = X_missing_columns[k_nearest_indices, :]
        weighted_average_of_neighboring_rows = dot(
            X_knn.T,
            neighbor_weights)
        sum_weights = neighbor_weights.sum()
        weighted_average_of_neighboring_rows /= sum_weights
        imputed_values = weighted_average_of_neighboring_rows

        observed_mask_missing_columns = observed_mask_column_major[:, missing_columns]
        observed_mask_missing_columns_sorted = observed_mask_missing_columns[
            neighbor_indices, :]

        # We can determine the maximum number of other rows that must be
        # inspected across all features missing for this row by
        # looking at the column-wise running sums of the observed feature
        # matrix.
        observed_cumulative_sum = observed_mask_missing_columns_sorted.cumsum(axis=0)
        sufficient_rows = (observed_cumulative_sum == k)
        n_rows_needed = sufficient_rows.argmax(axis=0) + 1
        max_rows_needed = n_rows_needed.max()

        if max_rows_needed == k:
            # if we never needed more than k rows then we're done after the
            # optimistic averaging above, so go on to the next sample
            X[i, missing_columns] = imputed_values
            continue

        # truncate all the sorted arrays to only include the necessary
        # number of rows (should significantly speed up the "slow" path)
        necessary_indices = neighbor_indices[:max_rows_needed]
        d_sorted = row_distances[necessary_indices]
        X_missing_columns_sorted = X_missing_columns[necessary_indices, :]
        observed_mask_missing_columns_sorted = observed_mask_missing_columns_sorted[
            :max_rows_needed, :]

        for missing_column_idx in range(n_missing_columns):
            # since all the arrays we're looking into have already been
            # sliced out at the missing features, we need to address these
            # features from 0..n_missing using missing_idx rather than j
            if n_rows_needed[missing_column_idx] == k:
                assert np.isfinite(imputed_values[missing_column_idx]), \
                    "Expected finite imputed value #%d (column #%d for row %d)" % (
                        missing_column_idx,
                        missing_columns[missing_column_idx],
                        i)
                continue
            row_mask = observed_mask_missing_columns_sorted[:, missing_column_idx]
            sorted_column_values = X_missing_columns_sorted[:, missing_column_idx]
            neighbor_distances = d_sorted[row_mask][:k]

            # may not have enough values in a column for all k neighbors
            k_or_less = len(neighbor_distances)
            usable_weights = neighbor_weights[:k_or_less]
            np.divide(
                1.0,
                neighbor_distances, out=usable_weights)
            neighbor_values = sorted_column_values[row_mask][:k_or_less]

            imputed_values[missing_column_idx] = (
                dot(neighbor_values, usable_weights) / usable_weights.sum())

        X[i, missing_columns] = imputed_values
    return X
