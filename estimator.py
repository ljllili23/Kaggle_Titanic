# -*- coding: utf-8 -*-
"""
author: LeeJiangLee
contact: ljllili23@gmail.com
2018.11.20 10:36 AM
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
clf = RandomForestClassifier(n_estimators=230,max_depth=2,random_state=0)

data_set = pd.read_csv('./data/train_one_hot_sex.csv')

print(data_set.dtypes)

train_data,test_data = train_test_split(data_set,random_state=0,train_size=0.8)
# print(train_data.shape,train_data.columns)
train_labels = train_data.Survived
# print(train_labels)
del train_data['Survived'],train_data['PassengerId']
#print(train_data.shape,train_data.columns,train_labels)
test_labels = test_data.Survived
del test_data['Survived'], test_data['PassengerId']

train_data = train_data.values
test_data = test_data.values
print(train_data.shape,test_data.shape)
clf.fit(train_data,train_labels)
i=1
count = 0
for data,label in zip(test_data,test_labels):
    data = np.array(data).reshape([1,-1])
    #print(data.shape)
    p_label = clf.predict(data).item()
    print("No.{0} prediction:{1}".format(i,p_label))
    print('ground truth:',label)
    if p_label == label:
        count += 1
    i += 1
print("the accuracy of the validation:{}".format(count/test_data.shape[0]))

