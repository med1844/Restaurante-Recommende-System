import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csc_matrix
from typing import Tuple, List, Any
from utils.data_utils import load_data
import os

class LoadData():
    PATH_SMALL = 'datasets/smallsets/'
    PATH_SAMPLE = 'datasets/samplesets/'

    def __init__(self, test_ratio=0.1):
        '''
        Intial the LoadData Class
        :param test_ratio: given test data ratio when loading data, default 0.1
        '''
        self.test_ratio = test_ratio

    def loadSmallDataSet(self, path=PATH_SMALL):
        '''
        Load the subsets of full Yelp restaurant dataset
        :param path: path of the dataset
        :return: train data and test data
        '''
        filenames = ("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json")

        # Load user, business, review subsets
        user_subset, business_subset, review_subset = load_data(path, filenames)
        df_user = pd.DataFrame(user_subset)
        df_business = pd.DataFrame(business_subset)
        df_review = pd.DataFrame(review_subset)
        
        n_users = df_review.user_id.unique().shape[0]
        n_items = df_review.business_id.unique().shape[0]
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))
        df_review['user_idx'] = df_review.groupby(['user_id']).ngroup()
        df_review['business_idx'] = df_review.groupby(['business_id']).ngroup()
        
        train_data, test_data = train_test_split(df_review, test_size=self.test_ratio)
        train_matrix = np.zeros((n_users, n_items))
        test_matrix = np.zeros((n_users, n_items))
        for line in train_data.itertuples():
          train_matrix[line[10], line[11]] = line[4]
        for line in test_data.itertuples():
          test_matrix[line[10], line[11]] = line[4]

        return train_matrix, test_matrix


    def loadSampleDataSet(self, path=PATH_SAMPLE):
        '''
        Load the sample Yelp restaurant dataset
        :param path: path of the dataset
        :return: train data and test data
        '''
        filenames = ("sample_users.json", "sample_business.json",
             "sample_reviews_train.json", "sample_reviews_test.json")

        # Load user, business, review subsets
        df_user = pd.read_json(os.path.join(path, filenames[0]))
        df_business = pd.read_json(os.path.join(path, filenames[1]))
        df_review_train = pd.read_json(os.path.join(path, filenames[2]))
        df_review_test = pd.read_json(os.path.join(path, filenames[3]))
        
        unique_user_ids = pd.concat([df_review_train['user_id'], df_review_test['user_id']]).unique()
        unique_business_ids = pd.concat([df_review_train['business_id'], df_review_test['business_id']]).unique()
        user_id_to_index = {user_id: index for index, user_id in enumerate(unique_user_ids, start=0)}
        business_id_to_index = {business_id: index for index, business_id in enumerate(unique_business_ids, start=0)}

        n_users = len(unique_user_ids)
        n_items= len(unique_business_ids)
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))

        df_review_train['user_idx'] = df_review_train['user_id'].map(user_id_to_index)
        df_review_test['user_idx'] = df_review_test['user_id'].map(user_id_to_index)
        df_review_train['business_idx'] = df_review_train['business_id'].map(business_id_to_index)
        df_review_test['business_idx'] = df_review_test['business_id'].map(business_id_to_index)
        
        train_matrix = np.zeros((n_users, n_items))
        test_matrix = np.zeros((n_users, n_items))
        for line in df_review_train.itertuples():
          train_matrix[line[10], line[11]] = line[4]
        for line in df_review_test.itertuples():
          test_matrix[line[10], line[11]] = line[4]

        return train_matrix, test_matrix
