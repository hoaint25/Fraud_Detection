##load the packages
from cProfile import label
from tkinter.ttk import LabeledScale
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
#iplot(fig, filename = 'class')

##DATA EXPLORATION

#transactions in time 
class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels =  ['Not Fraud','Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist = False, show_rug = False)
fig['layout'].update(title = 'Credit Card Transactions Time Density Plot', xaxis = dict(title='Time [s]'))
#iplot(fig, filename = 'dist_only')

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
#plt.show()

#plot the Number Transactions with Hour column per Class
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Transactions', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Transactions', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Total Number of Transactions by Hour (Not Fraud) ')
s2.set_title('Total Number of Transactions by Hour (Fraud) ')
#plt.show()

#plot the average Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Mean', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Mean', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Average Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Average Amount of Transactions by Hour (Fraud) ')
#lt.show()

#plot the maximum Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Max', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Max', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Maximum Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Maximum Amount of Transactions by Hour (Fraud) ')
#plt.show()

#plot the Median Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Median', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Median', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Median Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Median Amount of Transactions by Hour (Fraud) ')
#plt.show()

#plot the Minium Amount of Transactions
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(18,6))
s1 = sns.lineplot(ax=ax1, x = 'Hour', y = 'Min', data = df[df['Class'] == 0])
s2 = sns.lineplot(ax=ax2, x = 'Hour', y = 'Min', data = df[df['Class'] == 1], color = 'red')
s1.set_title('Min Amount of Transactions by Hour (Not Fraud) ')
s2.set_title('Min Amount of Transactions by Hour (Fraud) ')
#plt.show()

#transaction amount 
fig, (ax1,ax2) = plt.subplots(ncols=2, figsize = (12,6))
s1 = sns.boxplot(ax=ax1, x = "Class", y = "Amount", hue = "Class", data = data_df, palette = "PRGn", showfliers=True)
s2 = sns.boxplot(ax=ax2, x = "Class", y = "Amount", hue = "Class", data = data_df, palette = 'PRGn', showfliers = False)
#plt.show()

#describe about Fraud and Not Fraud Class
tmp = data_df[['Amount','Class']].copy()
class_0 = tmp[tmp['Class'] == 0]['Amount']
class_1 = tmp[tmp['Class'] == 1]['Amount']
print(class_0.describe())
print(class_1.describe())

#plot the fraudulent transactions (amount) against time. The time is shown is seconds form the start of the time period
fraud = data_df.loc[data_df['Class'] == 1]

trace = go.Scatter(
    x = fraud["Time"], y = fraud["Amount"],
    name = 'Amount',
    marker = dict(color ='rgb(238,23,11)', line = dict(color = 'red', width = 1), opacity = 0.5),
    text = fraud['Amount'],
    mode = 'markers')

data = trace
layout = dict(title = 'Amount of fraudulent transactions', 
            xaxis = dict(title = 'Time[s]', showticklabels = True),
            yaxis = dict(title = 'Amount'),
            hovermode = 'closest')

fig = dict(data = data, layout = layout)
#iplot(fig, filename = 'fraud-amount')

##Features corelation 
plt.figure(figsize = (14,4))
plt.title('Credit Card Transactions features correlation plot (Pearson)')
corr = data_df.corr()
sns.heatmap(corr, xticklabels=corr.columns, yticklabels = corr.columns, linewidths = 0.1, cmap = 'Reds')
#plt.show()

#plot the direct correlated value (V20 and Amount)
s = sns.lmplot(x = 'V20', y = 'Amount', data = data_df, hue = 'Class', fit_reg = True, scatter_kws = {'s':2})
plt.show()
#plot the direct correlated values (V7 and  Amount)
s = sns.lmplot(x = 'V7', y = 'Amount', data = data_df, hue = 'Class', fit_reg = True, scatter_kws = {'s':2})
#plt.show()

##Features density plot
var = data_df.columns.values
i = 0 
t0 = data_df[data_df['Class'] == 0]
t1 = data_df[data_df['Class'] == 1]

sns.set_style('whitegrid')
plt.figure()
fig, ax = plt.subplot(8,4, figsize(16,28))

for feature in var:
    sns.kdeplot(t0[feature], bw = 0.5, label = 'Class = 0')
    sns.kdeplot(t1[feature], bw = 0.5, label = 'Class = 0')
    plt.xlabel(feature, fontsize = 12)
    loc, labels = xticks
    plt.tick_params(axis = 'both', which = 'major', labelsize = 12)
plt.show()


##PREDICTIVE MODEL 

#Define target and predictor values
target = 'Class'
predictors = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',\
                'V11','V12','V13','V14','V15','V16','V17','V18','V19',\
                'V20','V21','V22','V23','V24','V25','V26','V7','V28',\
                'Amount']

#Split data in train, test and validation sets
train_df, test_df = train_test_split(data_df, test_size = 0.2, random_state = 2018, shuffle = True)
train_df, valid_df = train_test_split(data_df, test_size = 0.2, random_state = 2018, shuffle = True)

##RANDOM FORREST CLASSIFIER

#Define model parameters
clf = RandomForestClassifier(n_jobs = 4, random_state = 2018, criterion= 'gini', n_estimators=100, verbose = False)

#train the RandomForestClassifier using train_df and fit function 
clf.fit(train_df[predictors], train_df[target].values)

#predict the target values for the valid_df data, using predict function 
preds = clf.predict(valid_df[predictors])

#features importance 
tmp = pd.DataFrame({'Feature': predictors,'Feature Importance': clf.feature_importances_,})
tmp = tmp.sort_values(by = 'Feature Importance', ascending = False)
plt.figure(figsize = (7,4))
plt.title('Feature Importance', fontsize = 12)
s = sns.barplot(x='Feature', y = 'Feature Importance', data = tmp)
s.set_xticklabels(s.get_xticklabels(), rotation = 45)
plt.show()

print('Done')