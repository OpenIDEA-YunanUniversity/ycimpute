import numpy as np
from nose.tools import eq_

from imputer.knnimput import KNN


import pandas as pd
from datas import dpath

def testknn():
    data = pd.read_csv(dpath.test)
    data = np.asarray(data)
    fillX = KNN(k=3).complete(data)
    return fillX

if __name__=="__main__":
    testknn()
