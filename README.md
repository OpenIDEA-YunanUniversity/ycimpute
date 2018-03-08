
# ycimpute


![AppVeyor](https://img.shields.io/appveyor/ci/gruntjs/grunt.svg)
![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/Django.svg)

## [中文文档]( https://hcmy.gitbooks.io/ycimpute/content/)     [English Documentation](https://github.com/HCMY/ycimpute/blob/master/doc_eng.md)
# What is ycimpute?
ycimpute is a high-level API for padding missing values library. It is written in python, which integrates methods for missing values imputation based on machine learning and statistics. Some modules require scikit-lean support.

The original intention of writing this library is that I often encounter some missing values in the process of doing data mining, most of the missing values of the scene can use the same set of missing approach, so the final decision to write a function library to facilitate the call

## Up untill now, There are couple of methods I've been implementing:

For various algorithms' detail, Please look up the API below:

- simple imputation methods(mean value, padding zero, select maxmum, minimum ...etc)
- based on Random Forest (IterForest)
- Multiple Imputation(MICE)
- based on Expectation Maximization (EM)
- based on KNN
### Suggestion: Data loss mechanism varies in different scenarios, which requires the engineer to choose the appropriate filling method based on the business.
## Missing values can be of three general types:

>Missing Completely At Random (MCAR):
When missing data are MCAR, the presence/absence of data is completely independent of observable variables and parameters of interest. In this case, the analysis performed on the data are unbiased. In practice, it is highly unlikely.
>Missing At Random (MAR):
When missing data is not random but can be totally related to a variable where there is complete information. An example is that males are less likely to fill in a depression survey but this has nothing to do with their level of depression, after accounting for maleness. This kind of missing data can induce a bias in your analysis especially if it unbalances your data because of many missing values in a certain category.
>Missing Not At Random (MNAR):
When the missing values are neither MCAR nor MAR. In the previous example that would be the case if people tended not to answer the survey depending on their depression level.
Let's check out the performance of per imputation methods in various data sets:

### the data sets include: IRIS dataset WINE dataset Boston dataset.

## These are the complete data. I used them to experiment and evaluate the model after randomly deleting the data. About 10% of the data is missing, and each feature contains different degrees of data loss.

## All of the data are continuous, the evaluation function which I used was RMSE(root mean square error) Red line represents the average of all errors.
![葡萄酒数据集](https://github.com/HCMY/ycimpute/blob/master/img/WINE.svg)
![IRIS数据集](https://github.com/HCMY/ycimpute/blob/master/img/IRIS.svg)
![波士顿房产数据集](https://github.com/HCMY/ycimpute/blob/master/img/BOSTON.svg)


