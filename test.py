from sklearn.datasets import load_breast_cancer
from ycimpute.imputer import iterforest
import numpy as np
import pandas as pd

path = '/Users/ytu-egemen-zeytinci/Downloads/train.csv'
data = pd.read_csv(path).copy()

cols = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked']
df = data.drop(cols, axis=1).dropna()

fake = df.copy()
np.random.seed(100)
mask = np.random.choice([True, False], size=fake['Sex'].shape)
fake['Sex'] = fake['Sex'].mask(mask)

fake['Sex'] = fake['Sex'].replace({'female': 0, 'male': 1})

for col in fake.columns:
    if fake[col].nunique() < 10:
        print(f'{col}: {fake[col].unique()}')

X = np.array(fake)
dff = iterforest.IterImput().complete(X)
filled = pd.DataFrame(dff, columns=fake.columns)

for col in filled.columns:
    if filled[col].nunique() < 10:
        print(f'{col}: {filled[col].unique()}')
