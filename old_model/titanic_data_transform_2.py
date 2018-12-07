# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

19/11/2018 9:00 PM
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import OneHotEncoder
#using one hot encoding in pandas
clf = RandomForestClassifier(n_estimators=230,max_depth=2)

train_data = pd.read_csv('./data/train_transformed.csv')
labels = train_data.Survived
print(labels.shape)
print(train_data.columns)
train_data = pd.get_dummies(train_data, prefix=['sex'], columns=['Sex'])
# del train_data['Name'],train_data['Ticket'],train_data['Cabin'],train_data['Embarked']

print(train_data.columns)
train_data.to_csv('./data/train_one_hot_sex.csv',index=False)