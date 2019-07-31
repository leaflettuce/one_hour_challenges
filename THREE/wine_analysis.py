# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:13:48 2019

@author: andyj
"""

import pandas as pd
import numpy as np
import os

##########################
# import and set up data #
##########################

# setwd
os.chdir('E:/projects/one_hour_challenges/THREE')
# import
df = pd.read_csv('../data/wine_reviews.csv', index_col = 0)

# understand df
pd.set_option('display.max_columns', None)
df.shape
df.describe()
df.head()
df.info()
df.columns
df.values

# remove NA's
df.isnull().sum(axis = 0)

df = df[np.isfinite(df['price'])]
df = df[np.isfinite(df['points'])]

