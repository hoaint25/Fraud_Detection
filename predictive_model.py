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
#iplot(fig, filename = 'class')

##DATA EXPLORATION

#transactions in time 
class_0 = data_df.loc[data_df['Class'] == 0]["Time"]
class_1 = data_df.loc[data_df['Class'] == 1]["Time"]

hist_data = [class_0, class_1]
group_labels =  ['Not Fraud','Fraud']

fig = ff.create_distplot(hist_data, group_labels, show_hist = False, show_rug = False)
fig['layout'].update(title = 'Credit Card Transactions Time Density Plot', xaxis = dict(title='Time [s]'))
iplot(fig, filename = 'dist_only')
