import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.sparse import csc_matrix
from typing import Tuple, List, Any
from utils.data_utils import load_data

class LoadData():
    PATH_SMALL = 'datasets/subsets/'
    PATH_FULL= 'datasets/fullsets/'

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

    def loadFullDataSet(self, path=PATH_FULL):
        '''
        Load the full Yelp restaurant dataset
        :param path: path of the dataset
        :return: train data and test data
        '''
        filenames = ("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json")

        # Load user, business, review subsets
        user_fullset, business_fullset, review_fullset = load_data(path, filenames)
        df_user = pd.DataFrame(user_fullset)
        df_business = pd.DataFrame(business_fullset)
        df_review = pd.DataFrame(review_fullset)
        
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
