##load the packages
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
#init_notebook_mode(connected=True)

import gc
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn import svm
#import lightgbm as lgb
#from lightgbm import LGBMClassifier
import xgboost as xgb
import os

pd.set_option('display.max_columns', 100)

#read the data data
data_df = pd.read_csv("creditcard.csv")

#check the data 
print('Number of rows: ', data_df.shape[0])
print('Number of Columns: ', data_df.shape[1])

#glimpse the data
print(data_df.head())

#more detail about data 
print(data_df.describe())

#check the missing data 
total = data_df.isnull().sum().sort_values(ascending=False)
percent = (data_df.isnull().sum()/data_df.isnull().count()*100).sort_values(ascending=False)
pd.concat([total,percent], axis = 1, keys = ['Total','Percent']).transpose()

#data unbalanced
#check the data unbalance with respect with target value (class)
temp = data_df['Class'].value_counts()
df = pd.DataFrame({'Class':temp.index, 'Values': temp.values})

trace = go.Bar(x = df['Class'], y = df['Values'],
        name = 'Credit Card Fraud Class - data unbalance (Not fraud = 0, Fraud = 1',
        marker = dict(color='red'), text = df['Values'])

data = trace 
layout = dict(title = 'Credit Card Fraud Detection - data unbalance ( Not Fraud: 0, Fraud: 1',
                xaxis = dict(title = 'Class', showticklabels = True),
                yaxis = dict(title = 'Number of Transactions'),
                hovermode = 'closest', width = 600)
#fig = dict(data = data, layout = layout)
iplot(fig, filename = 'class')

##DATA EXPLORATION

#transactions in time 
class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels =  ['Not Fraud','Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist = False, show_rug = False)
fig['layout'].update(title = 'Credit Card Transactions Time Density Plot', xaxis = dict(title='Time [s]'))
iplot(fig, filename = 'dist_only')

#more details to the time distribution of both classes transactions, as well as aggregated values of transaction count and amount, per hour. We assume that the time unit is second
data_df['Hour'] = data_df['Time'].apply(lambda x: np.floor(x/3600))

tmp = data_df.groupby(['Hour','Class'])['Amount'].aggregate(['min','max','count','sum','mean','median','var']).reset_index()
df = pd.DataFrame(tmp)
df.columns = ['Hour','Class','Min','Max','Transactions','Sum','Mean','Median','Var']
print(df.head(10))

#plot the Total Amount by Hour per class (Fraud and Not Fraud)
fig, (ax1, ax2) = plt.subplots(ncols=2, figsize = (18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Sum', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Sum', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Total Amount by Hour (Not Fraud) ')
s2.set_title('Total Amount by Hour (Fraud) ')
plt.show()

#plot the Number Transactions with Hour column per Class
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Transactions', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Transactions', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Total Number of Transactions by Hour (Not Fraud) ')
s2.set_title('Total Number of Transactions by Hour (Fraud) ')
plt.show()

#plot the average Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Mean', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Mean', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Average Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Average Amount of Transactions by Hour (Fraud) ')
plt.show()

#plot the maximum Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Max', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Max', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Maximum Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Maximum Amount of Transactions by Hour (Fraud) ')
plt.show()

#plot the Median Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Median', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Median', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Median Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Median Amount of Transactions by Hour (Fraud) ')
plt.show()
