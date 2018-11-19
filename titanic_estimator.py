# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

18/11/2018 8:00 PM
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.base import TransformerMixin
import pandas as pd

train_file = "./data/train.csv"
test_file = "./data/test.csv"

train_data = pd.read_csv(train_file)
del train_data['Ticket']
del train_data['Cabin']
del train_data['Name']
del train_data['Embarked']
# cabins = train_data['Cabin']
# print(train_data.dtypes)
# cabins = cabins.values.reshape(-1, 1)
# enc = OrdinalEncoder()
# cabins = enc.fit_transform(cabins)
# print(cabins[0:9], cabins.dtype, cabins.shape)
# print(cabins.value_counts())
# print(train_data.columns,train_data)
sex = train_data['Sex']
print(sex.value_counts())
class DataFrameImputer(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.fill = pd.Series([X[c].value_counts().index[0] if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],index=X.columns)
        # for c in X 得到的是X的列名
        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)
train_data = DataFrameImputer().fit_transform(train_data)

print(train_data['Sex'],train_data['Age'])

train_data.to_csv('./data/train_transformed.csv')