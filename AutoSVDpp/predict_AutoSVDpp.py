from models.AutoSVDpp import AutoSVDpp
from utils.YelpRecommendationGenerator import *

# load the pretrained AutoSVDpp model
autosvdpp = AutoSVDpp(path_of_feature_file="datasets/samplesets/restaurant_features_encoded.csv")
autosvdpp.load_model()

# construct the Yelp Recommender
yelp_recommender = YelpRecommendationGenerator(autosvdpp)
df_user, df_business, df_review_train, df_review_test= yelp_recommender.load_sample_dataset()
user_id, user_df = yelp_recommender.generate_sample_user()
sorted_df = yelp_recommender.predict_stars_for_sample_user(user_id)

print("\n", "====================== predicted ratings of sample user ======================", "\n")
print(sorted_df)

# generate top k recommendations
k=10
stars_df, recommendations_df = yelp_recommender.generate_top_k_recommendations(user_id=user_id,k=k)

print("\n", "============================ top", k ,"recommedation ============================", "\n")
print(recommendations_df)

# save the predicted stars and recommendations
yelp_recommender.save_recommendations_autosvdpp()