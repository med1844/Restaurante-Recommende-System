import pandas as pd
import numpy as np
import torch
from torch import nn
import json
from nvtabular.framework_utils.torch.layers import ConcatenatedEmbeddings


def read_files(user_file, restaurants_file, embedding_table_shapes_file, model_file):
  involved_users = pd.read_pickle(user_file)
  encoded_restaurants = pd.read_pickle(restaurants_file)
  embedding_table_shape = json.load(open(embedding_table_shapes_file))
  model = WideAndDeep(embedding_table_shapes, 14, 432, 9)
  model.load_state_dict(torch.load(model_file))
  return involved_users, encoded_restaurants, model


def recommend_topK(user_id, involved_users, encoded_restaurants, model, K=10):
  assert user_id in involved_users['user_id'].values
  user_data = involved_users[involved_users['user_id'] == user_id].drop(columns=['user_id'])
  assert user_data.isna().sum().sum() == 0
  assert encoded_restaurants.isna().sum().sum() == 0
  encoded_restaurants_ids = encoded_restaurants['business_id']
  encoded_business_mat = encoded_restaurants.drop(columns=['business_id'])
  user_and_business_mat = pd.concat([user_data, encoded_business_mat], axis=1)
  user_and_business_mat = user_and_business_mat.fillna(method="ffill")
  wide_features_val, sparse_features_val, dense_features_val = user_and_business_mat[wide_features].values, user_and_business_mat[sparse_features].values, user_and_business_mat[dense_features].values
  t_wide_features_val, t_sparse_features_val, t_dense_features_val = torch.tensor(wide_features_val).float(), torch.tensor(sparse_features_val).long(), torch.tensor(dense_features_val).float()
  with torch.no_grad():
    pred = model(t_wide_features_val, t_sparse_features_val, t_dense_features_val)
    top10_idx = torch.argsort(torch.squeeze(pred), descending=True)[:10]
    top10_idx = top10_idx.numpy()
    top10_business_ids = encoded_restaurants_ids.iloc[top10_idx]
    print(f"recommend 10 restaurants for user {user_id}")
    for i in range(len(top10_idx)):
      print(f"restaurant id is {top10_business_ids.values[i]}, predicted score is {torch.squeeze(pred)[top10_idx[i]]}")
  return top10_idx


