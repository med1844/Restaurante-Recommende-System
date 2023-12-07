from interface import RestaurantRecommenderInterface
import os
from .models.AutoSVD import AutoSVD
from .utils.YelpRecommendationGenerator import YelpRecommendationGenerator


# simply wrap around to build aggregators
class AutoSVDPPRecommender(RestaurantRecommenderInterface):
    def __init__(self, path_of_feature_file: str) -> None:
        self.auto_svd_pp = AutoSVD(path_of_feature_file)
        self.auto_svd_pp.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), "parameters/"))
        self.yelp_recommender = YelpRecommendationGenerator(self.auto_svd_pp)
        _ = self.yelp_recommender.load_small_dataset(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets/subsets/"))

    def fit(self):
        raise NotImplementedError()

    def predict(self, user_id: str, top_n: int = 5):
        user_df = self.yelp_recommender.get_user_df(user_id)
        self.yelp_recommender.predict_stars_for_sample_user(user_df)
        return self.yelp_recommender.generate_top_k_recommendations(top_n)
