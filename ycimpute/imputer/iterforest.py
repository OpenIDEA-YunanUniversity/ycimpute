from ..utils.tools import Solver
from ..utils import config
from ..esemble.random_forest import RegressionForest
from ..esemble.random_forest import ClassificationForest

#***********************************************

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier

import numpy as np



class IterImput(Solver):
    def __init__(self):
        self.coltype_dict = None
        self.mask_memo_dict = None
        self.sorted_col = None
        self.stop = False
        # notes: I didn't ues my random_forest as its model.
        # The reason is, at the common situation,my model does not perform as well as scikit-learn's
        # up until now, also I didn't use any of parameters in these model, which should be improved, of course.
        self.rf_reg = RandomForestRegressor()#RegressionForest()
        self.rf_cla = RandomForestClassifier()#ClassificationForest()
        self.imp_continuous_index = None
        self.imp_categorical_index = None

    def solve(self, X):
        """
        implementation this paper:
        you could see the principal of this method in the paper
        :param X: yor original data matrix
        :return:
        """
        self.mask_memo_dict = self.masker(X)
        self.sorted_col = self.sort_col(X)
        self.coltype_dict = self._judge_type(X)
        self.imp_continuous_index, self.imp_categorical_index = \
            self.get_type_index(self.mask_memo_dict[config.all], self.coltype_dict)

        init_fill = self.fill(X, self.mask_memo_dict[config.all], fill_method='mean')

        differ_categorical = float('inf')
        differ_continuous = float('inf')


        while self.stop is False:

            differ_categorical_old = differ_categorical
            differ_continuous_old = differ_continuous

            x_old_imp  = init_fill[self.mask_memo_dict[config.all]]


            x_new_imp = []

            for col in self.sorted_col:
                tmp = []
                if self.coltype_dict[col] is config.categotical:
                    model = self.rf_cla
                else:
                    model = self.rf_reg


                x_obs, y_obs, x_mis = self.split(init_fill, col, self.mask_memo_dict)
                model.fit(x_obs, y_obs)
                y_mis = model.predict(x_mis)
                for ele in y_mis:
                    tmp.append(ele)
                    x_new_imp.append(ele)
                init_fill[:,col][self.mask_memo_dict[col]] = tmp
            x_new_imp = np.asarray(x_new_imp)

            differ_continuous, differ_categorical = self._lose_func(x_new_imp, x_old_imp)
            if differ_continuous >= differ_continuous_old and differ_categorical >= differ_categorical_old:
                self.stop = True
        return init_fill




    def _lose_func(self, imp_new, imp_old):
        """
        Evaluation Method, mathematical concept are available at 'https://www.stu-zhouyc.com/iterForest/metrics'

        :param imputed_data_old: a dict like {'col name':[predicted value1,...],...}
                                        the dict contains original missing index which is part of the original data
                                        its the last estimated data
                                        accompany with brand-new imputed data, they are going to be evaluate.
        :return:
        """

        continuous_imp_new = imp_new[self.imp_continuous_index]
        continuous_imp_old = imp_old[self.imp_continuous_index]
        categorical_imp_new = imp_new[self.imp_categorical_index]
        categorical_imp_old = imp_new[self.imp_categorical_index]


        try:
            continuous_div = continuous_imp_new-continuous_imp_old
            continuous_div = continuous_div.dot(continuous_div)
            continuous_sum = continuous_imp_new.dot(continuous_imp_new)

            categorical_count = np.sum(categorical_imp_new==categorical_imp_old)
            categorical_var_len = len(categorical_imp_new)

        except:
            categorical_var_len = 0.01
            categorical_count = 0

            continuous_div = 0
            continuous_sum = 0.001

        if categorical_var_len is 0:
            categorical_differ = 0
        else:
            categorical_differ = categorical_count / categorical_var_len

        if continuous_sum is 0:
            continuous_differ = 0
        else:
            continuous_differ = continuous_div / continuous_sum
        return continuous_differ, categorical_differ









