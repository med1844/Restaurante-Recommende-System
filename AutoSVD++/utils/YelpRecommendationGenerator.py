from models.AutoSVD import AutoSVD
from utils.data_utils import load_data
import pandas as pd
import numpy as np
from random import randint
import os

class YelpRecommendationGenerator():
    PATH_SMALL = 'datasets/subsets/'
    PATH_FULL= 'datasets/fullsets/'

    def __init__(self, model):
        '''
        Intial the YelpRecommendationGenerator Class
        :model: the pretrained AutoSVD model
        '''
        self.model = model

    def load_small_dataset(self, path=PATH_SMALL):
        '''
        Load the subsets of full Yelp restaurant dataset
        :param path: path of the dataset
        :return: train data and test data
        '''
        filenames = ("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json")
        
        # Load and preprocess the business and review subsets
        user_subset, business_subset, review_subset = load_data(path, filenames)
        df_user = pd.DataFrame(user_subset)
        df_business = pd.DataFrame(business_subset)
        df_review = pd.DataFrame(review_subset)
        df_review['user_idx'] = df_review.groupby(['user_id']).ngroup()
        df_review['business_idx'] = df_review.groupby(['business_id']).ngroup()
        self.df_user = df_user
        self.df_business = df_business
        self.df_review = df_review

        n_users = df_review.user_id.unique().shape[0]
        n_items = df_review.business_id.unique().shape[0]
        self.n_users = n_users
        self.n_items = n_items
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))
        
        return df_business, df_review

    def load_full_dataset(self, path=PATH_FULL):
        '''
        Load the full Yelp restaurant dataset
        :param path: path of the dataset
        :return: business and review data frames
        '''
        filenames = ("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json")
        
        # Load and preprocess the business and review subsets
        user_subset, business_subset, review_subset = load_data(path, filenames)
        df_user = pd.DataFrame(user_subset)
        df_business = pd.DataFrame(business_subset)
        df_review = pd.DataFrame(review_subset)
        df_review['user_idx'] = df_review.groupby(['user_id']).ngroup()
        df_review['business_idx'] = df_review.groupby(['business_id']).ngroup()
        self.df_user = df_user
        self.df_business = df_business
        self.df_review = df_review

        n_users = df_review.user_id.unique().shape[0]
        n_items = df_review.business_id.unique().shape[0]
        self.n_users = n_users
        self.n_items = n_items
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))  
        
        return df_business, df_review

    def generate_sample_user(self):
        '''
        Randomly sample an user from the loaded dataset
        :return: dataframe that contains information for the sample user
        '''
        sample_user_idx = randint(0, self.n_users-1)
        sample_user_idx = 76
        sample_df = self.df_review.loc[self.df_review["user_idx"] == sample_user_idx]
        user_df = sample_df.groupby(['user_idx', 'business_idx']).agg({'stars': 'mean'}).reset_index()

        # print the sampled user's profile
        user_id = self.df_review[self.df_review['user_idx'] == sample_user_idx]['user_id'].iloc[0]
        user_row = self.df_user[self.df_user['user_id'] == user_id]
        print("\n", "============== predict ratings of user ", sample_user_idx, " ==============", "\n")
        print(user_row)

        return user_df

    def predict_stars_for_sample_user(self, user_df):
        '''
        Predict the ratings the user will give to each business
        :param user_df: dataframe that contains information for the sample user
        :return: dataframe that contains predicted ratings of the sample user to all businesses (in desceding order)
        '''
        avg_rating = self.model.average_rating
        u = user_df['user_idx'][0]
        b_u = self.model.B_U[u]
        U_u = self.model.U[:, u]
        item_features = self.model.readItemFeature()

        predicted_ratings = []
        print("\n", "========= predict user's rating of each business =========", "\n") 
        for i in range(0, self.n_items):
          b_i = self.model.B_I[i]
          V_i = self.model.V[:, i]
          beta = self.model.beta

          if hasattr(self.model, 'Y'):
            # AutoSVD++
            dot = np.dot(U_u.T, (V_i + beta * item_features[i]))
            predicted_rating = avg_rating + b_u + b_i + dot
          else:
            # AutoSVD
            predicted_rating = self.model.predict(avg_rating, b_u, b_i, U_u, V_i, beta * item_features[i])
          
          predicted_ratings.append(predicted_rating)
          print("business_idx =", i, " predicted_star =", predicted_rating)

        # construct df for predictions
        prediction_df= pd.DataFrame(predicted_ratings)
        prediction_df['business_idx'] = list(range(0, self.n_items))
        prediction_df = prediction_df.merge(self.df_review[['business_idx', 'business_id']], on='business_idx', how='left')
        prediction_df = prediction_df.drop_duplicates()
        prediction_df = prediction_df.rename(columns={0: "stars"})

        # sort the df in descending order of predicted ratings
        sorted_df = prediction_df.sort_values(by='stars', ascending=False)
        self.sorted_df = sorted_df

        return sorted_df

    def generate_top_k_recommendations(self, k=10):
        '''
        Generate the top k recommendations for the sample user
        :return: top k recommended restaurants
        '''
        recommendations_df = pd.DataFrame(columns=self.df_business.columns)

        for index, row in self.sorted_df.head(k).iterrows():
          business_id = row['business_id']
          new_row = self.df_business[self.df_business['business_id'] == business_id]
          recommendations_df = pd.concat([recommendations_df, new_row], ignore_index=True)
        self.recommendations_df = recommendations_df
        
        return recommendations_df

    def save_recommendations_autosvd(self, directory='predictions/'):
        # Save the ratings and recommendations dataframes to the file
        self.sorted_df.to_csv(os.path.join(directory, "sorted_df.csv"), index=False)
        self.recommendations_df.to_csv(os.path.join(directory, "recommendations_df.csv"), index=False)

    def load_recommendations_autosvd(self, directory='predictions/'):
        # Load the ratings and recommendations dataframes from the file
        sorted_df = pd.read_csv(os.path.join(directory, "sorted_df.csv"))
        recommendations_df = pd.read_csv(os.path.join(directory, "recommendations_df.csv"))
        
        return sorted_df, recommendations_df

    def save_recommendations_autosvdpp(self, directory='predictions/'):
        # Save the ratings and recommendations dataframes to the file
        self.sorted_df.to_csv(os.path.join(directory, "sorted_df_pp.csv"), index=False)
        self.recommendations_df.to_csv(os.path.join(directory, "recommendations_df_pp.csv"), index=False)

    def load_recommendations_autosvdpp(self, directory='predictions/'):
        # Load the ratings and recommendations dataframes from the file
        sorted_df = pd.read_csv(os.path.join(directory, "sorted_df_pp.csv"))
        recommendations_df = pd.read_csv(os.path.join(directory, "recommendations_df_pp.csv"))
        
        return sorted_df, recommendations_df

