from svd_sparse_cf import SVDSparseCollaborativeFilteringRecommender
from AutoSVDpp.auto_svd_pp import AutoSVDPPRecommender
from NeuralCF.neural_cf import NeuralCFRecommender
from data_utils import load_data
from random import choice, seed
from typing import List, Any, Tuple
from random_recommender import RandomRecommender
import json
import torch
import os


torch.cuda.is_available = lambda : False
seed(6220)


def split(data: List[Any], ratio: float) -> Tuple[List[Any], List[Any]]:
    num_element = int(len(data) * ratio)
    return data[:num_element], data[num_element:]


def main():
    # user = json.load(open("./data/sample_users.json", "r"))
    # business = json.load(open("./data/sample_business.json", "r"))
    # review_train = json.load(open("./data/sample_reviews_train.json", "r"))
    # review_test = json.load(open("./data/sample_reviews_test.json", "r"))

    # model = AutoSVDPPRecommender(os.path.abspath("./AutoSVDpp/datasets/subsets/restaurant_features_encoded.csv"))
    # model.predict("JyzLjUFEIW3epNlHI6Oa6Q")

    model = NeuralCFRecommender(
        "./data/sample_reviews_test.json",
        # "./AutoSVDpp/datasets/subsets/yelp_academic_dataset_review.json",
        "./NeuralCF/Torch-NCF/checkpoints/checkpoints_neumf_5k_epoch100_l2-0.0000001/pretrain_neumf_factor8neg4_Epoch100_HR0.7258_NDCG0.3565.model",
        "5k",
    )
    print(model.predict("fr1Hz2acAb3OaL3l6DyKNg"))

    # for r in review_test:
    #     print(model.predict(r["user_id"]))

    # model = SVDSparseCollaborativeFilteringRecommender(user, business, review_train, 5)
    # model.fit()
    # print("svd sparse, k = 5")
    # print("train rmse", model.eval(review_train))
    # print("test rmse", model.eval(review_test))

    # print("random")
    # model = RandomRecommender(business)
    # print("train rmse", model.eval(review_train))
    # print("test rmse", model.eval(review_test))


if __name__ == "__main__":
    main()
