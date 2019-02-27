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
	- Clusters do strongly correlate to age and income though. 

OVERALL: Semi-supervised learning via clustering demographic data fails to predict occupation.

TIME TAKEN: 62 minutes
```
