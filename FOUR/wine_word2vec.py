# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:42:50 2019

@author: andyj
"""

import pandas as pd
import numpy as np
import os
import re

from nltk.corpus import stopwords
from tqdm import tqdm
from gensim.models import Word2Vec 

from sklearn import manifold
import umap
import matplotlib.pyplot as plt
##########################
# import and set up data #
##########################

# setwd
os.chdir('E:/projects/one_hour_challenges/FOUR')
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



###########################
# process text for word2v #
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


# create list of wine descriptions 
desc_list = []

# populate the list with the product codes
for i in tqdm(df.index[0:30000]):
    temp = df[df.index == i]["desc_clean"].tolist()
    temp = str(temp).split(' ')
    desc_list.append(temp)
    
###############
# Train Model #
###############
# train word2vec model
model = Word2Vec(window = 10, sg = 1, hs = 0,
                 negative = 10, # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed = 42)

model.build_vocab(desc_list, progress_per=200)

model.train(desc_list, total_examples = model.corpus_count, 
            epochs=10, report_delay=1)

model.init_sims(replace=True)
print(model)


# extract all vectors
X = model[model.wv.vocab]
X.shape

# feature reduction into 2-d
tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
X_tsne = tsne.fit_transform(X)

#cluster_embedding = umap.UMAP(n_neighbors=30, min_dist=0.0,
#                              n_components=2, random_state=42).fit_transform(X)



# visualize
plt.figure(figsize=(10,9))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], s=3, cmap='Spectral')
plt.title('Wine Review Vocab Scatter (t-sne)')

###############
# Recommender #
###############
 # set into ine df
df_used = df[0:30000]


# takes in list and returns reduced vectors
def get_average_vectors(df_used = df_used):
     # create place holders
    df_used['tsne_x'] = 0
    df_used['tsne_y'] = 0
    vector = []
    i = 0
    loc = -1

    for i in tqdm(desc_list):
        loc += 1
        try:
            vector.append(model.wv[i])
        except KeyError:
            try:
                df_used = df_used.drop(df.index[loc])
            except KeyError:
                break
   
    # set df_used to proper len
    df_used = df_used[0:len(vector)]
    
    # loop through all vectors and average out into df vars
    for i in tqdm(range(0, len(vector))):
        vec_tsne = tsne.fit_transform(vector[i])
            # set to df
        df_used.iloc[i, 16] = np.mean(vec_tsne[:, 0])
        df_used.iloc[i, 17] = np.mean(vec_tsne[:, 1])
    
    return


# get vectors of wine reviews
get_average_vectors()