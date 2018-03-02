
# ycimpute

 #### ycimpute 是一个用于缺失值填充的函数库。它用python写成，里面集成了一些基于机器学习，统计的缺失值填充的方法。部分模块需要[scikit-lean](http://scikit-learn.org/stable/)的支持.
 #### 写这个函数库的初衷是我在做数据挖掘的过程中经常遇到一些缺失值，大部分场景下的缺失值都可以使用同一套缺失处理办法，最后决定写成一个函数库方便调用
 ### 建议：不同场景下的数据缺失机制不同，这需要工程师基于对业务选择合适的填充方法。

### 各算法的填充效果
![葡萄酒数据集](https://github.com/HCMY/ycimpute/tree/master/img/1.png)
![IRIS数据集](https://github.com/HCMY/ycimpute/tree/master/img/2.png)
![葡萄酒数据集](https://github.com/HCMY/ycimpute/tree/master/img/3.png)
# 下载

### 通过pip


```python
#pip install ycimpute
```



### 从源码


```python
# TODO
```

# API

## 选择有监督的填充方法

### 基于随机森林填充
理论请参看这篇论文：

### 例子


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


### 参数说明：
### 目前模型可用参数 0

## MICE 填充


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


### 参数说明：
### 目前模型可用参数 TODO

## KNN 填充


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
    Imputing row 1/506 with 0 missing, elapsed time: 0.142
    Imputing row 101/506 with 2 missing, elapsed time: 0.145
    Imputing row 201/506 with 2 missing, elapsed time: 0.147
    Imputing row 301/506 with 3 missing, elapsed time: 0.149
    Imputing row 401/506 with 1 missing, elapsed time: 0.151
    Imputing row 501/506 with 1 missing, elapsed time: 0.153
    X filled
    
     []


### 参数说明：
### 目前模型可用参数 KNN(parameters)
k, int

## 查看填充效果，评价函数采用rmse


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

    28.1530101564


### 你也可以一次性查看所有模型的填充效果
注意，所用模型的参数均为默认，有待改善


```python
from ycimpute.utils.test_evaluate import show
result = show.analysiser(X_missing,X_original)
import pandas as pd
result = pd.DataFrame.from_dict(result, orient='index')
print(result)
```

    Imputing row 1/506 with 0 missing, elapsed time: 0.055
    Imputing row 101/506 with 2 missing, elapsed time: 0.058
    Imputing row 201/506 with 2 missing, elapsed time: 0.060
    Imputing row 301/506 with 3 missing, elapsed time: 0.062
    Imputing row 401/506 with 1 missing, elapsed time: 0.065
    Imputing row 501/506 with 1 missing, elapsed time: 0.067
                                    0
    rmse_mice_score         28.962190
    rmse_iterforest_score   24.854965
    rmse_knn_score          40.944330
    rmse_mean_score         52.154860
    rmse_zero_score        159.534384
    rmse_median_score       57.616702
    rmse_min_score         127.874980

