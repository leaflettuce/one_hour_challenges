# One Hour Challenges
### Quick burst challenge projects
Different projects I attempt with a one hour time limit. I'll set either a goal or library/methodology I want to use and attempt to complete the project within an hour.


#### ONE - Semi Supervised Learning
```
GOAL: See if I can predict a person's occupation by different demographic statistics

METHOD: 
	- I use the audit data provided by Rattle to pull demographic and occupation data
	- Quick cleaning of the data into form suited for models to be used
	- Cluster the data using k-means 
	- Imputer clusters into dataset as additional input for inference
	- Classify using k-nn to predict occupation
	
RESULTS: 
	- Clustering results display defined segmentation between the give inputs (age, income, marital, gender).
	- Particularly defined groups at either 3, 5, and 7 k.
	- Using a k = sqrt(col_len) results in a 'decent' predictor for k-nn.
	- Too many occupation possibilities and results are overall weak.
	- On average accurate predicting an occupation at around ~25%

TIME TAKEN: 52 minutes
```
