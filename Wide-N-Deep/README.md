#	Wide and Deep Model

## What we've used and modified
We've implemented and trained: 
- Wide and Deep with a regression task

We've changed and added:
- Data: Subset of Yelp dataset
- Training: training.ipynb
- Inferencing code: `inference.py`
- When inferencing, user_file is /Wide-N-Deep/SavedFiles/sample_users.pkl
- When inferencing, model_path is /Wide-N-Deep/SavedFiles/model_27.pt
- When inferencing, restaurants_file is /Wide-N-Deep/SavedFiles/encoded_restaurants.pkl
- When inferencing, embedding_table_shapes_file is /Wide-N-Deep/SavedFiles/embedding_table_shapes.json


## Train and Test Data Statistics

RMSE of 1.0311 on training set, 1.0188 on testing set, RMSE of 1.0341 on testing set. 

NDCG of 0.9657 on training set and 0.9675 on testing set, NDCG of 0.9672 on testing set. 
