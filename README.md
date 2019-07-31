# One Hour Challenges
### Quick burst challenge projects
Different projects I attempt with a one hour time limit. I'll set either a goal or library/methodology I want to use and attempt to complete the project within an hour.


#### ONE - Semi-Supervised Learning
```
GOAL: Can I cluster people by their demographic statistics and predict occupation?

METHOD: 
	- I use the audit data provided by Rattle to pull demographic and occupation data
	- Quick cleaning of the data into form suited for models to be used
	- Cluster the data using k-means 
	- Train a model with clusters as target dependent
	- Classify using k-nn to predict cluster
	- Does cluster relate to occupation at all?
	
RESULTS: 
	- Clustering results display defined segmentation between the give inputs (age, income, marital, gender).
	- Particularly defined groups at either k = 3.
		-> clusters = (young and high income, low income and male, older and high income)
	- Using a k = 5 results in a great predictor for cluster classification
		-> 100% accuracy predicting cluster in test data. (too accurate? why?)
	- Clusters very weakly relate to occupations. Not a success
	- Clusters do strongly correlate to age and income though 

OVERALL: Semi-supervised learning via clustering demographic data fails to predict occupation.

TIME TAKEN: 62 minutes
```


#### TWO - EDA w/ Matplotlib and Seaborn
```
GOAL: How many insights can I discover from an hour of eda with matplotlib/seaborn?

METHOD: 
	- Data obtained from Kaggle dataset: 'Pokemon with Stats'
	- I typically run eda in R so wanted to get practice in Py
	- Pull in data and drop incomplete rows
	- Get descriptive stats of data
	- Run univariate analysis of data
	- Run bivariate analysis of data
	- Dive deeper into initial insights with multivariate analysis.
	- What are the most interesting trends to visualize more deeply?
	
RESULTS: 
	- Learned how to subplot in matplotlib
	- Learned about querying in pandas
	- Linear Relationships between most card stats
	- Indicating different power levels rather than trade-offs between stats
	- Water type seems to be on the fringes suggesting more volatile stats exchanges
	- Grass types seem to be more consistent yet average
	- Stat totals about evenly distributed between generations

OVERALL: Seaborn is a great tool for making quick and beautiful charts when conducting EDA.

TIME TAKEN: 76 minutes
```



#### THREE - Recommender System using TF-IDF
```
GOAL: Can we suggest similar wines given an input wine?

METHOD: 
	- Data obtained from Kaggle dataset: 'Wine Reviews'
	- Get some practice with NLP and create a recommender system
	- Pull in data and drop incomplete rows
	- Text preprocessing
	- Cosine distance similarity function
	- Suggest top 5 based on input
	
RESULTS: 
	- TBD

OVERALL: TBD

TIME TAKEN: TBD
```


#### FOUR - Identifying Similarity with Word2vec
```
GOAL: Can we map out similar wine describers in a vector space to highlight trends in flavors? 

METHOD: 
	- Data obtained from Kaggle dataset: 'Wine Reviews'
	- Get some practice with NLP and learn to apply word2vec 
	- Pull in data and drop incomplete rows
	- Text preprocessing
	- Cosine distance similarity function
	- Map out in vector space
	
RESULTS: 
	- TBD

OVERALL: TBD

TIME TAKEN: TBD
```