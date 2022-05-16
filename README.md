# Credit Card Fraud Predictive Detection Models
## Problem Statement 

We are moving towards digital word - cybersecurity is becoming a crucial part of our life. When we talk about the security, one of the most challenge is fraudelent credit card transactions.

In this project, we will analyse customer data which contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. Since the dataset is highly balanced, so it needs to be handled before model building.

## Business Problem Overview

When we make transaction while purchasing any product online, a good amount of people prefer credit cards. The credit 
limit in credit cards sometimes helps us me making purchases even if we don’t have the amount at that time. but, on the 
other hand, these features are misused by cyber attackers. To tackle this problem we need a system that can abort the 
transaction if it finds fishy.
Today, we have many machine learning alogrithms that help up classify abnormal transactions. The only requirement is 
past data and the suitable that can fit our data.
This model and data can help reduce time-consuminh manual reviews, costly chargebacks and fee, and denials of legitimate 
transactions.

## About Data 
The dataset can be downloaded using this [link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
The dataset includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a toatal of 284,807 transactions, 492 were fraudulent. This dataset is highly unbalanced, with the positive class(frauds) accounting for 0.172% of the total transactions. 
- The dataset has been modified with Pricipals Components Analysis (PCA) to maintain confidentially. 
- Feature Time contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature Amount is the transaction Amount, this feature can be used for example-dependant cost-senstive learning.
- Apart from 'time' and 'amount', all the other features (V1,V2,V,V4 up to V28) are the principle components abtained using PCA. The feature 'amount' is the transactions amount. 
- The feature 'class' represent class labeling, and it takes the value 1 in case of fraud and 0 in others.
