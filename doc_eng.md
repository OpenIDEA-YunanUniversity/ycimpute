
### Welcome to ycimpute!

## ycimpute Overview
ycimpute is a high-level API for padding missing values library. It is written in python, which integrates methods for missing values imputation based on machine learning and statistics. Some modules require scikit-lean support.
The original intention of writing this library is that I often encounter some missing values in the process of doing data mining, most of the missing values of the scene can use the same set of missing approach, so the final decision to write a function library to facilitate the call
Suggestion: Data loss mechanism varies in different scenarios, which requires the engineer to choose the appropriate filling method based on the business.

## performence of various models

![UCI WINE data set](https://github.com/HCMY/ycimpute/blob/master/img/WINE.svg)
![IRIS data set](https://github.com/HCMY/ycimpute/blob/master/img/IRIS.svg)
![BOSTON housing data set](https://github.com/HCMY/ycimpute/blob/master/img/BOSTON.svg)

# Install

### via pip

pip install ycimpute

### via source


```python
git clone https://github.com/HCMY/ycimpute.git
cd ycimpute
python setup install
```

## API Reference

## select surpvised methods

### 1 based on Random Forest

 theories of this method: [MissForest—non-parametric missing value imputation for mixed-type data](https://academic.oup.com/bioinformatics/article/28/1/112/219101)

### usage:

#### Before using the example, you need to download the data file and copy it to the function directory of your python package 
(  ``` your python path / site-packages / ycimpute / datasets / ``` ) 
#### Linux users can use wget download, data download in the current working directory:
 ```
 wget https://github.com/HCMY/ycimpute/raw/master/test_data/boston.hdf5
 wget https://github.com/HCMY/ycimpute/raw/master/test_data/iris.hdf5
 wget https://github.com/HCMY/ycimpute/raw/master/test_data/wine.hdf5
 ```


```python
import numpy as np
from ycimpute.datasets.load_data import load_boston
from ycimpute.imputer.iterforest import IterImput
X_missing, X_original = load_boston()#加载boston房产数据

print(X_missing.shape)
print("X missing\n\n",np.argwhere(np.isnan(X_missing)))
X_filled = IterImput().complete(X_missing)
print("X filled\n\n",np.argwhere(np.isnan(X_filled)))
```

    (506, 13)
    X missing
    
     [[  1   2]
     [  1   4]
     [  1   8]
     ..., 
     [502  12]
     [504   3]
     [504   7]]
    X filled
    
     []


### parameters：
### TODO

## fill based on MICE 
theories of this method：[Multiple Imputation by Chained Equations](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/)

#### usage: 


```python
from ycimpute.imputer.mice import MICE
print("X missing\n\n",np.argwhere(np.isnan(X_missing)))
X_filled = MICE().complete(X_missing)
print("X filled\n\n",np.argwhere(np.isnan(X_filled)))
```

    X missing
    
     [[  1   2]
     [  1   4]
     [  1   8]
     ..., 
     [502  12]
     [504   3]
     [504   7]]
    X filled
    
     []


### parameters：
### TODO

## select unsurpvised methods

### based on KNN

#### usage


```python
from ycimpute.imputer.knnimput import KNN
print("X missing\n\n",np.argwhere(np.isnan(X_missing)))
X_filled = KNN(k=4).complete(X_missing)
print("X filled\n\n",np.argwhere(np.isnan(X_filled)))
```

    X missing
    
     [[  1   2]
     [  1   4]
     [  1   8]
     ..., 
     [502  12]
     [504   3]
     [504   7]]
    Imputing row 1/506 with 0 missing, elapsed time: 0.094
    Imputing row 101/506 with 2 missing, elapsed time: 0.096
    Imputing row 201/506 with 2 missing, elapsed time: 0.098
    Imputing row 301/506 with 3 missing, elapsed time: 0.100
    Imputing row 401/506 with 1 missing, elapsed time: 0.102
    Imputing row 501/506 with 1 missing, elapsed time: 0.104
    X filled
    
     []


### parameters
parameter | function | value 
- | :-: | -: 
k | --- | int 

## visualization fill effects of these method，metrics used by rmse


```python
from ycimpute.utils.tools import Solver
from ycimpute.utils import evaluate
from ycimpute.datasets.load_data import load_boston
solver = Solver()
X_missing, X_original = load_boston()
from ycimpute.imputer.mice import MICE

X_filled = MICE().complete(X_missing)
mask_all = solver.masker(X_missing)['all']
missing_index = evaluate.get_missing_index(mask_all)
original_arr = X_original[missing_index]
mice_filled_arr = X_filled[missing_index]
rmse_mice_score = evaluate.RMSE(original_arr, mice_filled_arr)
print(rmse_mice_score)
```

    29.1028614966


### you could look over all of methods effects one shot:
 notes: all the model use default parameters, which shoule be improved :)


```python
from ycimpute.utils.test_evaluate import show
result = show.analysiser(X_missing,X_original)
import pandas as pd
result = pd.DataFrame.from_dict(result, orient='index')
print(result)
```

    Imputing row 1/506 with 0 missing, elapsed time: 0.050
    Imputing row 101/506 with 2 missing, elapsed time: 0.052
    Imputing row 201/506 with 2 missing, elapsed time: 0.054
    Imputing row 301/506 with 3 missing, elapsed time: 0.056
    Imputing row 401/506 with 1 missing, elapsed time: 0.058
    Imputing row 501/506 with 1 missing, elapsed time: 0.060
                                    0
    rmse_mice_score         28.971895
    rmse_iterforest_score   23.639840
    rmse_knn_score          40.944330
    rmse_mean_score         52.154860
    rmse_zero_score        159.534384
    rmse_median_score       57.616702
    rmse_min_score         127.874980
