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
import matplotlib.pyplot as plt
import seaborn as sns
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
# SET UP REC  #
###############
 # set into ine df
df_used = df[0:30000]

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
# df_used = df_used[0:len(vector)]

# loop through all vectors and average out into df vars
for i in tqdm(range(1, len(vector))):
    vec_tsne = tsne.fit_transform(vector[i])
        # set to df
    df_used.iloc[i, 14] = np.mean(vec_tsne[:, 0])
    df_used.iloc[i, 15] = np.mean(vec_tsne[:, 1])


###########################
# VIZ SIMILARITY in WINES #
###########################
    
plt.figure(figsize=(10,9))
sns.lmplot(data=df_used, x='tsne_x', y='tsne_y', hue='variety', 
                   fit_reg=False, legend=True)
plt.title('Wine Similarity Based Upon Description')

#######################
# RECOMMENDER FINALLY #
#######################
    
    # for testing
df_used = df_used[0:6454]
wines = df_used[["title", "description"]]

# create wine-ID and wine-description dictionary
wines_dict = wines.groupby('title')['description'].apply(list).to_dict()

# get similar wines    
def similar_products(input_desc, n = 6):
    # extract most similar products for the input vector
    #input_desc = input_desc.split(' ')
    ms = model.similar_by_vector(input_desc, topn= n+1)[1:]

    # extract name and similarity score of the similar products
    new_ms = []
    for m in ms:
        for i,j in wines_dict.items():
            if m[0] in str(j):
                new_ms.append(i)
    
    return new_ms[0:10]     

similar_products('fruity')

# formatting into reusable function

def get_recommendations(test_input, df_used = df_used):
    test_split = test_input.split(' ')
    
    test_m = model.wv[test_split]
    test_tsne = tsne.fit_transform(test_m)
    
    test_x = np.mean(test_tsne[:, 0])
    test_y = np.mean(test_tsne[:, 1])
    
    df_results = {}
    for i in range (0, len(df_used)):
        wine_x = df_used.iloc[i, 14]
        wine_y = df_used.iloc[i, 15]
        X = test_x - wine_x
        Y = test_y - wine_y
        distance = np.sqrt((X*X) + (Y*Y))
        df_results[df_used.iloc[i, 10]] = distance
        
    sorted_results = sorted(df_results.items(), key=lambda kv: kv[1])
    
    return sorted_results
    

user_input = 'red cherry sour'
sorted_results = get_recommendations(user_input)
for i in range(0, 6):
    print(sorted_results[i][0], ', ', sorted_results[i][1])