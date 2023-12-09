from interface import RestaurantRecommenderInterface
import os
from .models.AutoSVD import AutoSVD
from .utils.YelpRecommendationGenerator import YelpRecommendationGenerator
from typing import Dict, List, Optional


# simply wrap around to build aggregators
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
