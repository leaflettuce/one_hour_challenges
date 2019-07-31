# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:13:48 2019

@author: andyj
"""

import pandas as pd
import os

##########################
# import and set up data #
##########################

# setwd
os.chdir('E:/projects/one_hour_challenges/THREE')
# import
df = pd.read_csv('../data/wine_reviews.csv')