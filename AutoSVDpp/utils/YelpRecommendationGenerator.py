from ..models.AutoSVD import AutoSVD
from data_utils import load_data
import pandas as pd
import numpy as np
from random import randint
import os
import json
from result import Ok


class YelpRecommendationGenerator:
    PATH_SMALL = "datasets/smallsets/"
    PATH_SAMPLE = "datasets/samplesets/"

    def __init__(self, model):
        """
        Intial the YelpRecommendationGenerator Class
        :model: the pretrained AutoSVD model
        """
        self.model = model

    def load_small_dataset(
        self,
        path=PATH_SMALL,
        filenames=(
            "yelp_academic_dataset_user.json",
            "yelp_academic_dataset_business.json",
            "yelp_academic_dataset_review.json",
        ),
    ):
        """
        Load the subsets of full Yelp restaurant dataset
        :param path: path of the dataset
        :return: train data and test data
        """

        # Load and preprocess the business and review subsets
        user_subset, business_subset, review_subset = load_data(
            path, filenames, line=False
        )

        df_user = pd.DataFrame(user_subset)
        df_business = pd.DataFrame(business_subset)
        df_review = pd.DataFrame(review_subset)
        df_review["user_idx"] = df_review.groupby(["user_id"]).ngroup()
        df_review["business_idx"] = df_review.groupby(["business_id"]).ngroup()
        self.df_user = df_user
        self.df_business = df_business
        self.df_review = df_review

        n_users = df_review.user_id.unique().shape[0]
        n_items = df_review.business_id.unique().shape[0]
        self.n_users = n_users
        self.n_items = n_items
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))

        return df_business, df_review

    def load_sample_dataset(self, path=PATH_SAMPLE):
        """
        Load the sample Yelp restaurant subset
        :param path: path of the dataset
        :return: business and review data frames
        """
        filenames = (
            "sample_users.json",
            "sample_business.json",
            "sample_reviews_train.json",
            "sample_reviews_test.json",
        )

        # Load and preprocess the business and review subsets
        df_user = pd.read_json(os.path.join(path, filenames[0]))
        df_business = pd.read_json(os.path.join(path, filenames[1]))
        df_review_train = pd.read_json(os.path.join(path, filenames[2]))
        df_review_test = pd.read_json(os.path.join(path, filenames[3]))

        self.df_user = df_user
        self.df_business = df_business
        self.df_review_train = df_review_train
        self.df_review_test = df_review_test

        unique_user_ids = pd.concat(
            [df_review_train["user_id"], df_review_test["user_id"]]
        ).unique()
        unique_business_ids = pd.concat(
            [df_review_train["business_id"], df_review_test["business_id"]]
        ).unique()
        user_id_to_index = {
            user_id: index for index, user_id in enumerate(unique_user_ids, start=0)
        }
        business_id_to_index = {
            business_id: index
            for index, business_id in enumerate(unique_business_ids, start=0)
        }

        self.user_id_to_index = user_id_to_index
        self.business_id_to_index = business_id_to_index

        df_review_train["user_idx"] = df_review_train["user_id"].map(user_id_to_index)
        df_review_test["user_idx"] = df_review_test["user_id"].map(user_id_to_index)
        df_review_train["business_idx"] = df_review_train["business_id"].map(
            business_id_to_index
        )
        df_review_test["business_idx"] = df_review_test["business_id"].map(
            business_id_to_index
        )

        n_users = len(unique_user_ids)
        n_items = len(unique_business_ids)
        self.n_users = n_users
        self.n_items = n_items
        print("Number of users=" + str(n_users) + "; Number of items=" + str(n_items))

        return df_user, df_business, df_review_train, df_review_test

    def get_user_df(self, user_id: str):
        # Find the user_idx for the given user_id
        user_idx = self.df_review[self.df_review["user_id"] == user_id][
            "user_idx"
        ].iloc[0]

        # Filter df_review using the user_idx to create user_df
        user_df = self.df_review[self.df_review["user_idx"] == user_idx]
        user_df = (
            user_df.groupby(["user_idx", "business_idx"])
            .agg({"stars": "mean"})
            .reset_index()
        )

        return user_df

    def generate_sample_user(self):
        """
                Randomly sample an user from the loaded dataset
                :return: dataframe that contains information for the sample user
        <<<<<<< HEAD
        """
        sample_user_idx = randint(0, self.n_users - 1)
        sample_user_idx = 76
        sample_df = self.df_review.loc[self.df_review["user_idx"] == sample_user_idx]
        user_df = (
            sample_df.groupby(["user_idx", "business_idx"])
            .agg({"stars": "mean"})
            .reset_index()
        )

        # print the sampled user's profile
        user_id = self.df_review[self.df_review["user_idx"] == sample_user_idx][
            "user_id"
        ].iloc[0]
        user_row = self.df_user[self.df_user["user_id"] == user_id]
        print(
            "\n",
            "============== predict ratings of user ",
            sample_user_idx,
            " ==============",
            "\n",
        )
        print(user_row)

        return user_id, user_df

    def predict_stars_for_sample_user(self, user_id):
        """
        Predict the ratings the user will give to each business
        :param user_df: dataframe that contains information for the sample user
        :return: dataframe that contains predicted ratings of the sample user to all businesses (in desceding order)
        """
        avg_rating = self.model.average_rating

        u = self.user_id_to_index[user_id]
        b_u = self.model.B_U[u]
        U_u = self.model.U[:, u]
        item_features = self.model.readItemFeature()

        predicted_ratings = []

        # print("\n", "========= predict user's rating of each business =========", "\n")
        for i in range(0, self.n_items):
            b_i = self.model.B_I[i]
            V_i = self.model.V[:, i]
            beta = self.model.beta

            if hasattr(self.model, "Y"):
                # AutoSVD++
                dot = np.dot(U_u.T, (V_i + beta * item_features[i]))
                predicted_rating = avg_rating + b_u + b_i + dot
            else:
                # AutoSVD
                predicted_rating = self.model.predict(
                    avg_rating, b_u, b_i, U_u, V_i, beta * item_features[i]
                )

            predicted_ratings.append(predicted_rating)
        # print("business_idx =", i, " predicted_star =", predicted_rating)

        # Invert the business_map dictionary to map indices to business IDs
        index_to_business_id = {v: k for k, v in self.business_id_to_index.items()}
        # Create a DataFrame from the inverted dictionary
        df = pd.DataFrame(
            list(index_to_business_id.items()), columns=["business_idx", "business_id"]
        )
        # Add the ratings to the DataFrame
        df["stars"] = df["business_idx"].apply(lambda x: predicted_ratings[x])
        # Sort the DataFrame based on ratings in descending order
        sorted_df = df.sort_values(by="stars", ascending=False)
        self.sorted_df = sorted_df

        return sorted_df

    def predict_stars_for_all_users(self):
        test_matrix = np.zeros((self.n_users, self.n_items))

        avg_rating = self.model.average_rating
        item_features = self.model.readItemFeature()
        beta = self.model.beta

        for u in range(test_matrix.shape[0]):  # Iterate over users
            b_u = self.model.B_U[u]
            U_u = self.model.U[:, u]
            for i in range(test_matrix.shape[1]):  # Iterate over items
                b_i = self.model.B_I[i]
                V_i = self.model.V[:, i]
                if hasattr(self.model, "Y"):
                    # AutoSVD++
                    dot = np.dot(U_u.T, (V_i + beta * item_features[i]))
                    predicted_rating = avg_rating + b_u + b_i + dot
                else:
                    # AutoSVD
                    predicted_rating = self.model.predict(
                        avg_rating, b_u, b_i, U_u, V_i, beta * item_features[i]
                    )
                test_matrix[u][i] = predicted_rating

        return test_matrix

    def generate_top_k_recommendations(self, user_id, business_ids=None, k=10):
        """
        Generate the top k recommendations for the given user_id among given business_ids
        :return: top k recommended restaurants
        """
        if business_ids == None:
            business_ids = list(self.business_id_to_index.keys())
        sorted_df = self.predict_stars_for_sample_user(user_id=user_id)
        recommendations_df = pd.DataFrame(columns=self.df_business.columns)
        stars_df = pd.DataFrame(columns=self.sorted_df.columns)

        i = 0
        for index, row in sorted_df.iterrows():
            if i == k:
                break
            business_id = row["business_id"]
            if business_id not in business_ids:
                continue
            new_row = self.df_business[self.df_business["business_id"] == business_id]
            recommendations_df = pd.concat(
                [recommendations_df, new_row], ignore_index=True
            )
            stars_df = pd.concat([stars_df, pd.DataFrame([row])], ignore_index=True)
            i = i + 1
        self.recommendations_df = recommendations_df

        merged_df = pd.merge(
            stars_df, recommendations_df[["business_id", "name"]], on="business_id"
        )
        result_list = [
            (row["name"], row["business_id"], row["stars"])
            for index, row in merged_df.iterrows()
        ]

        return Ok(result_list)

    def save_recommendations_autosvd(self, directory="predictions/"):
        # Save the ratings and recommendations dataframes to the file
        self.sorted_df.to_csv(os.path.join(directory, "sorted_df.csv"), index=False)
        self.recommendations_df.to_csv(
            os.path.join(directory, "recommendations_df.csv"), index=False
        )

    def load_recommendations_autosvd(self, directory="predictions/"):
        # Load the ratings and recommendations dataframes from the file
        sorted_df = pd.read_csv(os.path.join(directory, "sorted_df.csv"))
        recommendations_df = pd.read_csv(
            os.path.join(directory, "recommendations_df.csv")
        )

        return sorted_df, recommendations_df

    def save_recommendations_autosvdpp(self, directory="predictions/"):
        # Save the ratings and recommendations dataframes to the file
        self.sorted_df.to_csv(os.path.join(directory, "sorted_df_pp.csv"), index=False)
        self.recommendations_df.to_csv(
            os.path.join(directory, "recommendations_df_pp.csv"), index=False
        )

    def load_recommendations_autosvdpp(self, directory="predictions/"):
        # Load the ratings and recommendations dataframes from the file
        sorted_df = pd.read_csv(os.path.join(directory, "sorted_df_pp.csv"))
        recommendations_df = pd.read_csv(
            os.path.join(directory, "recommendations_df_pp.csv")
        )

        return sorted_df, recommendations_df
