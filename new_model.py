# -*- coding: utf-8 -*-
"""
author:  LeeJiangLee
contact: ljllili23@gmail.com

22/11/2018 9:23 PM
"""
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns

data_raw = pd.read_csv('./data/train.csv')
data_val = pd.read_csv('./data/test.csv')
data1 = data_raw.copy(deep=True)
data_cleaner = [data1,data_val]


# print(data_raw.info())
#
# print(data_raw.sample(10))
#
# print('Train columns with null values:\n',data1.isnull().sum())
# print("-"*10)
# print('Test/Validation columns with null values:\n',data_val.isnull().sum())
# print("-"*10)
# print(data_raw.describe(include='all'))


# CLEAN DATA
for dataset in data_cleaner:
    dataset['Age'].fillna(dataset['Age'].median(),inplace=True)
    dataset['Embarked'].fillna(dataset['Embarked'].mode()[0],inplace=True)  # mode() return items share the maximum frequency
    # print(dataset['Age'].median())
    dataset['Fare'].fillna(dataset['Fare'].median(),inplace=True)

drop_column = ['PassengerId','Cabin','Ticket']
data1.drop(drop_column,axis=1,inplace=True)

# print(data1.isnull().sum())
# print("-"*10)
# print(data_val.isnull().sum())
# print('\n',dataset['PassengerId'].loc[dataset['SibSp']>1])
for dataset in data_cleaner:
    dataset['FamilySize'] = dataset['SibSp']+dataset['Parch'] + 1
    dataset['IsAlone'] = 1
    dataset['IsAlone'].loc[dataset['FamilySize']>1] = 0
    dataset['Title'] = dataset['Name'].str.split(',',expand=True)[1].str.split('.',expand=True)[0]
    # qcut and cut
    dataset['FareBin'] = pd.qcut(dataset['Fare'],4)
    dataset['AgeBin'] = pd.cut(dataset['Age'].astype(int),5)
# print(data1.Title.value_counts())

stat_min = 10

title_names = (data1.Title.value_counts()<stat_min)

data1.Title = data1.Title.apply(lambda x: 'Misc' if title_names.loc[x]==True else x)
print('-'*10)
print(data1.Title.value_counts())
print('-'*10)

# Convert Formats

label = LabelEncoder()
for dataset in data_cleaner:
    dataset['Sex_Code'] = label.fit_transform(dataset.Sex)
    dataset['Embarked_Code'] = label.fit_transform(dataset.Embarked)
    dataset['Title_Code'] = label.fit_transform(dataset.Title)
    dataset['AgeBin_Code'] = label.fit_transform(dataset.AgeBin)
    dataset['FareBin_Code'] = label.fit_transform(dataset.FareBin)

Target = ['Survived']
data1_x = ['Sex','Pclass','Embarked','Title','SibSp','Parch','Age','Fare','FamilySize','IsAlone']
data1_x_calc = ['Sex_Code','Pclass','Embarked_Code','Title_Code','SibSp','Parch','Age','Fare']
data1_xy = Target + data1_x
# print('Original X Y:', data1_xy,'\n')
data1_x_bin = ['Sex_Code','Pclass','Embarked_Code','Title_Code','FamilySize','AgeBin_Code','FareBin_Code']
data1_xy_bin = Target + data1_x_bin
# print("Bin X Y:",data1_xy_bin, '\n')

data1_dummy = pd.get_dummies(data1[data1_x])
# print(data1_dummy)
data1_x_dummy = data1_dummy.columns.tolist()
data1_xy_dummy = Target + data1_x_dummy
# print("Dummy X Y:",data1_xy_dummy,'\n')


# Split Training and Testing Data

train1_x,test1_x,train1_y,test1_y = model_selection.train_test_split(data1[data1_x_calc],data1[Target],random_state=0)
train1_x_bin,test1_x_bin,train1_y_bin,test1_y_bin = model_selection.train_test_split(data1[data1_x_bin],data1[Target],random_state=0)
train1_x_dummy,test1_x_dummy,train1_y_dummy,test1_y_dummy = model_selection.train_test_split(data1_dummy[data1_x_dummy],data1[Target],random_state=0)

print("Data1 Shape: {}".format(data1.shape))
print("Train1 Shape: {}".format(train1_x.shape))
print("Test1 Shape: {}".format(test1_x.shape))

print("*"*10,'\n',train1_x_bin.head())

for x in data1_x:
    if data1[x].dtype != 'float64':
        print('-' * 10, '\n')
        print('Survival Correlation by:',x)
        print(data1[[x,Target[0]]].groupby(x,as_index=False).mean())
        print('-'*10,'\n')

plt.figure(figsize=[16,12])
plt.subplot(231)
plt.boxplot(x=data1['Fare'],showmeans=True,meanline=True)
plt.title('Fare Boxplot')
plt.ylabel('Fare ($)')