

from imputer.simple import SimpleFill


import pandas as pd
import numpy as np
from datas import dpath


data = pd.read_csv(dpath.test)
data = np.asarray(data)

def mean_value_imp():
    result = SimpleFill().complete(data)
    print(result)

def median_value_imp():
    result = SimpleFill(fill_method='median').complete(data)
    print(result)

def min_value_imp():
    result = SimpleFill(fill_method='min').complete(data)
    print(result)

def random_value_imp():
    result = SimpleFill(fill_method='random').complete(data)
    print(result)


def zero_value_imp():
    result = SimpleFill(fill_method='zero').complete(data)
    print(result)


if __name__=="__main__":
    mean_value_imp()
    median_value_imp()
    min_value_imp()
    random_value_imp()
    zero_value_imp()




