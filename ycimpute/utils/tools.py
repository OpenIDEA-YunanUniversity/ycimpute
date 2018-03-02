# -*- coding: utf-8 -*-



import numpy as np

from ..utils import config

class Solver(object):
    def __init__(
            self,
            fill_method: object = "zero",
            n_imputations = 1,
            min_value: object = None,
            max_value: object = None,
            normalizer: object = None) -> object:
        self.fill_method = fill_method
        self.n_imputations = n_imputations
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer

    def __repr__(self):
        return str(self)

    def __str__(self):
        field_list = []
        for (k, v) in sorted(self.__dict__.items()):
            if v is None or isinstance(v, (float, int)):
                field_list.append("%s=%s" % (k, v))
            elif isinstance(v, str):
                field_list.append("%s='%s'" % (k, v))
        return "%s(%s)" % (
            self.__class__.__name__,
            ", ".join(field_list))

    def _check_input(self, X):
        if len(X.shape) != 2:  # Note that ndarray's shpe is a tuple like (rows, cols)
            raise ValueError("Expected 2d matrix, got %s array" % (X.shape,))


    def _check_missing_value_mask(self, missing):
        """
        check whether your wait-imputation data contains null value
        :param missing: missing totally as your 'mask', an numpy array see above.
        :return:raise error
        """
        if not missing.any():
            raise ValueError("Input matrix is not missing any values")
        if missing.all():
            raise ValueError("Input matrix must have some non-missing values")

    def _judge_type(self, X):
        coltype_dic = {}
        for col in range(X.shape[1]):
            col_val = X[:, col]
            nan_index = np.where(np.isnan(col_val))
            col_val = np.delete(col_val, nan_index)
            #len(np.unique(col_val)) <= 2 or
            if (col_val.dtype.kind !='f') and (np.any(col_val == col_val.astype(int))):
                coltype_dic[col] = config.categotical
            else:
                coltype_dic[col] = config.continuous
        return coltype_dic

    def sort_col(self, X):
        """
        count various cols, the missing value wages,
        :param X: the original data matrix which is waiting to be imputed
        :return: col1, col2,.... colx, those cols has been sorted according its status of missing values
        """
        mask = np.isnan(X)
        nan_index = np.where(mask == True)[1]
        unique = np.unique(nan_index)
        nan_index = list(nan_index)
        dict = {}
        for item in unique:
            count = nan_index.count(item)
            dict[item] = count
        tmp = sorted(dict.items(), key=lambda e: e[1], reverse=True)
        sort_index = []
        for item in tmp:
            sort_index.append(item[0])
        return sort_index


    def get_type_index(self, mask_all, col_type_dict):
        """
        get the index of every missing value, because the imputed array is 1D
        where the continuous and categorical index are needed.

        :param mask_all:
        :param col_type_dict:
        :return: double list
        """
        where_target = np.argwhere(mask_all == True)
        imp_categorical_index = []
        imp_continuous_index = []
        index = 0
        for col in where_target[:, 1]:
            col_type = col_type_dict[col]
            if col_type is config.categotical:
                imp_categorical_index.append(index)
            elif col_type is config.continuous:
                imp_continuous_index.append(index)
            index += 1

        return imp_continuous_index, imp_categorical_index

    def masker(self, X):
        """
        find various columns missing val and complete missing matrix(bool)
        :param X: original data
        :return: a dict like {col:bool,..., 'all':bool matrix}
        """
        mask_rember = {}
        for col in range(X.shape[1]):
            col_val = X[:, col]
            mask_rember[col] = np.isnan(col_val)
        mask_rember[config.all] = np.isnan(X)
        return mask_rember

    @staticmethod
    def _fill_columns_with_fn(X, missing_mask, method):
        """

        :param X: numpy array, the data which waiting to be imputation
        :param missing_mask:numpy array
        :param method: the way of what kind of normal imputation algorithm you use
        :return:
        """
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()  # np.sum() which could calculate the number of 'TRUE'
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = method(col_data)
            X[missing_col, col_idx] = fill_values

    def fill(self, X, missing_mask, fill_method=None, inplace=False):
        """
        Parameters
        ----------
        X : np.array or pandas.DataFrame
            Data array containing NaN entries

        missing_mask : np.array
            Boolean array indicating where NaN entries are
            matrix like: [[T,F,T T],
                          [F,T,T,T]
                          [.......]]

        fill_method : str
            "zero": fill missing entries with zeros
            "mean": fill with column means
            "median" : fill with column medians
            "min": fill with min value per column
            "random": fill with gaussian samples according to mean/std of column

        inplace : bool
            Modify matrix or fill a copy
        """
        if not inplace:
            X = X.copy()

        if not fill_method:
            fill_method = self.fill_method

        if fill_method not in ("zero", "mean", "median", "min", "random"):
            raise ValueError("Invalid fill method: '%s'" % (fill_method))
        elif fill_method == "zero":
            # replace NaN's with 0
            X[missing_mask] = 0  # this is the match data feature of numpy array
        elif fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        # elif fill_method == "random":
        #    self._fill_columns_with_fn(
        #        X,
        #        missing_mask,
        #        col_fn=generate_random_column_samples)
        return X

    def prepare_input_data(self, X):
        """
        Check to make sure that the input matrix and its mask of missing
        values are valid. Returns X and missing mask.
        """
        X = np.asarray(X)
        if X.dtype != "f" and X.dtype != "d":
            X = X.astype(float)

        self._check_input(X)
        missing_mask = np.isnan(X)
        self._check_missing_value_mask(missing_mask)
        return X, missing_mask

    def split(self, X, target_col, mask):
        col_mask = mask[target_col]
        nan_index = np.where(col_mask == True)
        not_nan_index = np.where(col_mask == False)

        contain_nan_rows = np.delete(X, not_nan_index, 0)
        no_contain_nan_rows = np.delete(X, nan_index, 0)

        train_X = np.delete(no_contain_nan_rows, target_col, 1)
        train_y = no_contain_nan_rows[:, target_col]
        test_X = np.delete(contain_nan_rows, target_col, 1)

        return train_X, train_y, test_X

    def clip(self, X, col_type_dict):
        """
        Clip values to fall within any global or column-wise min/max constraints
        """


        for col in range(X.shape[1]):
            col_type = col_type_dict[col]
            if col_type is config.categotical:
                X[:,col][X[:, col]>0.5] = 1
                X[:, col][X[:, col] < 0.5 ] = 0
        return X


    def solve(self, X):
        """
        Given an initialized matrix X and a mask of where its missing values
        had been, return a completion of X.
        """
        raise ValueError("%s.solve not yet implemented!" % (
            self.__class__.__name__,))


    def complete(self, X):
        """
        Expects 2d float matrix with NaN entries signifying missing values

        Returns completed matrix without any NaNs.
        """
        self._check_input(X)
        self._check_missing_value_mask(np.isnan(X))
        col_type_dict = self._judge_type(X)
        X = self.solve(X)
        return self.clip(X, col_type_dict)

    @staticmethod
    def _get_missing_loc(missing_mask):
        missing_tuple = np.where(missing_mask)
        missing_row = missing_tuple[0]
        missing_col = missing_tuple[1]
        location = zip(missing_row, missing_col)
        return location, missing_row, missing_col

    @staticmethod
    def _pure_data(data, missing_mask):
        """
        pure a completely data set from data
        :param data: a matrix which contains missing value
        :param missing_mask:
        :return: a complete data set
        """
        missing_rows = np.where(missing_mask)[0]
        pure_data = np.delete(data, missing_rows, axis=0)

        return pure_data

    def _is_mix_type(self, X):
        mask_dict = self.masker(X)
        categorical_count = 0
        continuous_count = 0
        for col in range(X.shape[1]):
            col_type = mask_dict[col]
            if col_type is config.categotical:
                categorical_count+=1
            elif col_type is config.continuous:
                continuous_count+=1

        if categorical_count ==0 and continuous_count !=0:
            return config.continuous
        elif categorical_count!=0 and continuous_count ==0:
            return config.categotical
        elif categorical_count!=0 and continuous_count !=0:
            return config.mix
        else:
            raise ("unkonwn col type")

    @staticmethod
    def extract_imp_data_by_col(filled_X, required_cols, col_mask):
        x_imp = []
        for col in required_cols:
            nan_val = filled_X[:, col][col_mask[col]]
            for item in nan_val:
                x_imp.append(item)
        return x_imp
