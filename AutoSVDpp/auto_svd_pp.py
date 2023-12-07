from interface import RestaurantRecommenderInterface


from .models.AutoSVD import AutoSVD
from .utils.YelpRecommendationGenerator import YelpRecommendationGenerator


# simply wrap around to build aggregators
class AutoSVDPPRecommender(RestaurantRecommenderInterface):
    def __init__(self, path_of_feature_file: str) -> None:
        self.auto_svd_pp = AutoSVD(path_of_feature_file)
        self.yelp_recommender = YelpRecommendationGenerator(self.auto_svd_pp)
        _ = self.yelp_recommender.load_small_dataset()
        # user_df = yelp_recommender.generate_sample_user()
        # sorted_df = yelp_recommender.predict_stars_for_sample_user(user_df)

    def fit(self):
        raise NotImplementedError()

    def predict(self, user_id: str, top_n: int = 5):
        self.yelp_recommender.predict_stars_for_sample_user
