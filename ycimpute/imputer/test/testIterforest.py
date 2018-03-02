# -*- coding: utf-8 -*-



from utils.tools import Solver
from datas import dpath
import pandas as pd
import numpy as np



from sklearn.datasets import load_iris
#data = pd.read_csv(dpath.iris_test)
#imputed_data = IterForestInput(original_data=data).response()
#print(imputed_data)

from imputer.iterforest import IterImput
def test_iterimpute():
    data = pd.read_csv(dpath.iris_test)
    data = np.asarray(data)
    #data = load_iris().data
    print(data)
    filled = IterImput().complete(data)
    print(filled)


if __name__=="__main__":
    test_iterimpute()
