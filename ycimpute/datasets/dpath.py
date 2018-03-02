# -*- coding: utf-8 -*-

import h5py



def make_missing(npdata):
    import random
    import numpy as np
    rows, cols = npdata.shape
    random_cols = range(cols)
    for col in random_cols:
        random_rows = random.sample(range(rows - 1), int(0.1 * rows))
        npdata[random_rows, col] = np.nan
    return npdata


def create_data(data):
    import copy
    full_data = copy.copy(data)
    missing_data = make_missing(data)

    return missing_data, full_data


def boston():
    from sklearn.datasets import load_boston
    boston = load_boston()
    data = boston.data
    missing_data, full_data = create_data(data)
    h5_file = h5py.File('boston.hdf5','w')
    h5_file['missing'] = missing_data
    h5_file['full'] = full_data
    h5_file.close()


def diabetes():
    """
    Pima Indians Diabetes Datase
    :return:
    """
    from sklearn.datasets import load_diabetes
    load_diabetes = load_diabetes()
    data = load_diabetes.data
    missing_data, full_data = create_data(data)
    h5_file = h5py.File('diabetes.hdf5', 'w')
    h5_file['missing'] = missing_data
    h5_file['full'] = full_data
    h5_file.close()


def iris():
    from sklearn.datasets import load_iris
    data = load_iris().data
    missing_data, full_data = create_data(data)
    h5_file = h5py.File('iris.hdf5', 'w')
    h5_file['missing'] = missing_data
    h5_file['full'] = full_data
    h5_file.close()

def wine():
    from sklearn.datasets import load_wine
    data = load_wine().data
    missing_data, full_data = create_data(data)
    h5_file = h5py.File('wine.hdf5', 'w')
    h5_file['missing'] = missing_data
    h5_file['full'] = full_data
    h5_file.close()

if __name__=="__main__":
    #boston()
    #diabetes()
    #iris()
    wine()