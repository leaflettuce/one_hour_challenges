# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 07:26:24 2019

@author: andyj
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os 

########################
# contacts epxloration #
########################

# setwd
os.chdir('E:/projects/one_hour_challenges/TWO')
# import
df = pd.read_csv('../data/pokemon.csv', encoding = "ISO-8859-1")

# understand df
df.shape
df.describe()
df.head()
df.info()
df.columns

# drop id row
df = df.drop('#', axis = 1)

# check for missing
df.isnull().sum()
             
# create feature Is two Types
def set_dual_type(row):
    if type(row['Type 2']) == str:
        return 1
    else:
        return 0
    
df['dual_type'] = df.apply(lambda row: bool(set_dual_type(row)), axis=1)       
        
# Univariate Plots
sns.distplot(df['HP'], norm_hist=False, kde=True, bins=30).set(xlabel='HP', ylabel='Count')
sns.distplot(df['Attack'], norm_hist=False, kde=True, bins=30).set(xlabel='Atk', ylabel='Count')
sns.distplot(df['Defense'], norm_hist=False, kde=True, bins=30).set(xlabel='Dfn', ylabel='Count')
sns.distplot(df['Speed'], norm_hist=False, kde=True, bins=30).set(xlabel='Spd', ylabel='Count')
