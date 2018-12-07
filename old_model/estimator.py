# -*- coding: utf-8 -*-
"""
author: LeeJiangLee
contact: ljllili23@gmail.com
2018-11-20 11:30 am
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
from titanic_data_transform import DataFrameImputer

clf = RandomForestClassifier(n_estimators=230,max_depth=2,random_state=0)

train_data = pd.read_csv('./data/train_one_hot_sex.csv')
test_data = pd.read_csv('./data/test.csv')

# test_data transformation
del test_data['PassengerId'],test_data['Name'],test_data['Ticket'],test_data['Cabin'],test_data['Embarked']

test_data = DataFrameImputer().fit_transform(test_data)
test_data = pd.get_dummies(test_data,prefix=['sex'],columns=['Sex'])
print(test_data.columns)
train_labels = train_data.Survived
del train_data['Survived'],train_data['PassengerId']
# test_labels = test_data.Survived


train_data = train_data.values
test_data = test_data.values

clf.fit(train_data,train_labels)
i=1
count = 0
p_labels_array = []

for i,data in enumerate(test_data):
    data = np.array(data).reshape([1,-1])
    p_label = clf.predict(data).item()
    p_labels_array.append(p_label)
    #print("No.{0} prediction:{1}".format(i,p_label))
    # print('ground truth:',label)

# print("the accuracy of the validation:{}".format(count/test_data.shape[0]))
print(len(p_labels_array))
with open('solution.csv','w') as csvfile:
    fields = ['PassengerId','Survived']
    writer = csv.DictWriter(csvfile,fieldnames=fields)
    writer.writeheader()
    for i in range(len(p_labels_array)):
        if(i%100 == 0):
            print("writing:{}".format(i/1000))
        writer.writerow({'PassengerId':str(i+892),'Survived':p_labels_array[i]})

print("finished!!!")