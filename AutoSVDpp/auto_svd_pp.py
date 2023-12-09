from interface import RestaurantRecommenderInterface
import os
from .models.AutoSVD import AutoSVD
from .utils.YelpRecommendationGenerator import YelpRecommendationGenerator
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from result import Result, Ok


# simply wrap around to build aggregators
class AutoSVDPPMatrixRecommender(RestaurantRecommenderInterface):
    def __init__(self) -> None:
        dataset_path = os.path.abspath("data")
        filenames = (
            "sample_users.json",
            "sample_business.json",
            "sample_reviews_train.json",
            "sample_reviews_test.json",
        )
        df_user = pd.read_json(os.path.join(dataset_path, filenames[0]))
        df_business = pd.read_json(os.path.join(dataset_path, filenames[1]))
        df_review_train = pd.read_json(os.path.join(dataset_path, filenames[2]))
        df_review_test = pd.read_json(os.path.join(dataset_path, filenames[3]))

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
        self.matrix = np.load("test_matrix_pp.npy")

    def fit(self):
        pass

    def predict(
        self, user_id: str, business_ids: Optional[List[str]] = None, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        u_i = self.user_id_to_index[user_id]
        if business_ids is None:
            business_ids = list(self.business_id_to_index.keys())
        b_is = list(map(lambda i: self.business_id_to_index[i], business_ids))
        res = []
        for b_i, business_id in zip(b_is, business_ids):
            business_name = self.df_business[
                self.df_business["business_id"] == business_id
            ]["name"].values[0]
            rating = self.matrix[u_i][b_i]
            res.append((business_name, business_id, rating))

        return Ok(res)


class AutoSVDPPRecommender(RestaurantRecommenderInterface):
    def __init__(self, path_of_feature_file: str) -> None:
        self.auto_svd_pp = AutoSVD(path_of_feature_file)
        self.auto_svd_pp.load_model(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters/")
        )
        self.yelp_recommender = YelpRecommendationGenerator(self.auto_svd_pp)
        _ = self.yelp_recommender.load_sample_dataset(
            os.path.abspath("data"),
        )

    def fit(self):
        pass

    def predict(
        self, user_id: str, business_ids: Optional[List[str]] = None, top_n: int = 5
    ):
        # user_df = self.yelp_recommender.get_user_df(user_id)
        self.yelp_recommender.predict_stars_for_sample_user(user_id)
        return self.yelp_recommender.generate_top_k_recommendations(
            user_id, business_ids, k=top_n
        )
