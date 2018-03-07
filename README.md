
# ycimpute


![AppVeyor](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg)
![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)

## [中文文档]( https://hcmy.gitbooks.io/ycimpute/content/)     [English Documentation](https://github.com/HCMY/ycimpute/blob/master/doc_eng.md)
 #### ycimpute 是一个用于缺失值填充的函数库。它用python写成，里面集成了一些基于机器学习，统计的缺失值填充的方法。部分模块需要[scikit-lean](http://scikit-learn.org/stable/)的支持.
 #### 写这个函数库的初衷是我在做数据挖掘的过程中经常遇到一些缺失值，大部分场景下的缺失值都可以使用同一套缺失处理办法，最后决定写成一个函数库方便调用
 ### 建议：不同场景下的数据缺失机制不同，这需要工程师基于对业务选择合适的填充方法。

### 各算法的填充效果
![葡萄酒数据集](https://github.com/HCMY/ycimpute/blob/master/img/WINE.svg)
![IRIS数据集](https://github.com/HCMY/ycimpute/blob/master/img/IRIS.svg)
![波士顿房产数据集](https://github.com/HCMY/ycimpute/blob/master/img/BOSTON.svg)


