# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 10:13:48 2019

@author: andyj
"""

import pandas as pd
import numpy as np
import os
import re

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
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

# test description
df.iloc[0,1]


#######
# EDA #
#######

# QUICK EDA - Thanks to Susan Li at Data Science Central for the top_n_words function!

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
# run function
common_words = get_top_n_words(df['description'], 20)
df1 = pd.DataFrame(common_words, columns = ['desc' , 'count'])

# plot
plt.figure(figsize=(20,10))
plt.bar(df1['desc'], df1['count'])
plt.ylabel('count')
plt.title('Top 20 words in wine description (stop words removed)')


# BIGRAMS
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range =(2,2), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# run function
common_words = get_top_n_bigram(df['description'], 20)
df2 = pd.DataFrame(common_words, columns = ['desc' , 'count'])

# plot
plt.figure(figsize=(20,10))
plt.bar(df2['desc'], df2['count'])
plt.ylabel('count')
plt.xticks(rotation=90)
plt.title('Top 20 words in wine description (BIGRAM, stop words removed)')

# TRIGRAMS
def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range =(3,3), stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

# run function
common_words = get_top_n_trigram(df['description'], 20)
df3 = pd.DataFrame(common_words, columns = ['desc' , 'count'])

# plot
plt.figure(figsize=(20,10))
plt.bar(df3['desc'], df3['count'])
plt.ylabel('count') 
plt.xticks(rotation=45)
plt.title('Top 20 words in wine description (TRIGRAM, stop words removed)')


# get description lengths
df['word_count'] = df['description'].apply(lambda x: len(str(x).split()))
plt.hist(df['word_count'], bins=range(min(df['word_count']), max(df['word_count']) + 2, 2))
plt.title('Description Length Histogram')
plt.ylabel('Count')
plt.xlabel('Words in Description')


###########################
# process text for td-idf #
###########################
# Again thanks to Susan Li @ DSC
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text): # remove symbols, numbers, stopwords, and set to lower case!
    text = text.lower() # set to lower case
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwords from text
    return text
    
df['desc_clean'] = df['description'].apply(clean_text)

################
# TD_IDF Model #
################
# minor cleaning and cut df size for memory issues
df.set_index('title', inplace = True)
df_cut = df[0:25000]
# gtet tf-idf scores
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df_cut['desc_clean'])
# find cosine similiarity scores
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) # get cosine distance from A and B

indices = pd.Series(df.index)

def recommendations(name, cosine_similarities = cosine_similarities):
    recommended_wines = []
    
    # gettin the index of the hotel that matches the name
    idx = indices[indices == name].index[0]

    # creating a Series with the similarity scores in descending order
    score_series = pd.Series(cosine_similarities[idx]).sort_values(ascending = False)

    # getting the indexes of the 10 most similar hotels except itself
    top_10_indexes = list(score_series.iloc[1:11].index)
    
    # populating the list with the names of the top 10 matching hotels
    for i in top_10_indexes:
        recommended_wines.append(list(df.index)[i])
        
    return recommended_wines

###########
# test it #
###########


recommendations('Rainstorm 2013 Pinot Gris (Willamette Valley)')

desc_original = df_cut[df_cut.index == 'Rainstorm 2013 Pinot Gris (Willamette Valley)']['description'][0]
desc_one = df_cut[df_cut.index == 'Finca Jakue 2015 White (Getariako Txakolina)']['description'][0]


recommendations('Darioush 2013 Merlot (Napa Valley)')

desc_original_2 = df_cut[df_cut.index == 'Darioush 2013 Merlot (Napa Valley)']['description'][0]
desc_one_2 = df_cut[df_cut.index == 'Rodney Strong 2013 Zinfandel (Dry Creek Valley)']['description'][0]