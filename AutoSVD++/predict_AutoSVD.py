from models.AutoSVD import AutoSVD
from utils.YelpRecommendationGenerator import *

# load the pretrained AutoSVD model
autosvd = AutoSVD(path_of_feature_file="datasets/subsets/restaurant_features_encoded.csv")
autosvd.load_model()

# construct the Yelp Recommender
yelp_recommender = YelpRecommendationGenerator(autosvd)
df_business, df_review = yelp_recommender.load_small_dataset()
user_df = yelp_recommender.generate_sample_user()
sorted_df = yelp_recommender.predict_stars_for_sample_user(user_df)

print("\n", "============ predicted ratings of sample user ============", "\n")
print(sorted_df)

# generate top k recommendations
k=10
recommendations_df = yelp_recommender.generate_top_k_recommendations(k)

print("\n", "================== top", k ,"recommedation ==================", "\n")
print(recommendations_df)

# save the predicted stars and recommendations
yelp_recommender.save_recommendations_autosvd()