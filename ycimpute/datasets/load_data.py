import h5py
from os.path import dirname

import numpy as  np

def load_iris():
    abs_path = dirname(__file__).__add__('/iris.hdf5')
    try:
        file = h5py.File(abs_path,'r')
        missing_x = file['missing']
        original_x = file['full']
        return np.asarray(missing_x), np.asarray(original_x)
    except:
        file.close()
        raise ("can't load data")

def load_boston():
    abs_path = dirname(__file__).__add__('/boston.hdf5')
    try:
        file = h5py.File(abs_path,'r')
        missing_x = file['missing']
        original_x = file['full']
        return np.asarray(missing_x), np.asarray(original_x)
    except:
        file.close()
        raise ("can't load data")



def load_wine():
    abs_path = dirname(__file__).__add__('/wine.hdf5')
    try:
        file = h5py.File(abs_path,'r')
        missing_x = file['missing']
        original_x = file['full']
        return np.asarray(missing_x), np.asarray(original_x)
    except:
        file.close()
        raise ("can't load data")