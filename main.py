from svd_sparse_cf import SVDSparseCollaborativeFilteringRecommender
from AutoSVDpp.auto_svd_pp import AutoSVDPPRecommender
from NeuralCF.neural_cf import NeuralCFRecommender
from data_utils import load_data
from random import choice, seed
from typing import List, Any, Tuple
from random_recommender import RandomRecommender
from ensemble import RMSEWeightedEnsembler
from result import Ok, Err
import json
import torch
import os
import numpy as np


torch.cuda.is_available = lambda: False
seed(6220)


def split(data: List[Any], ratio: float) -> Tuple[List[Any], List[Any]]:
    num_element = int(len(data) * ratio)
    return data[:num_element], data[num_element:]


def main():
    user = json.load(open("./data/sample_users.json", "r"))
    business = json.load(open("./data/sample_business.json", "r"))
    review_train = json.load(open("./data/sample_reviews_train.json", "r"))
    review_test = json.load(open("./data/sample_reviews_test.json", "r"))

    # model = AutoSVDPPRecommender(
    #     os.path.abspath("./AutoSVDpp/datasets/subsets/restaurant_features_encoded.csv")
    # )
    # model.predict("JyzLjUFEIW3epNlHI6Oa6Q")

    # model = NeuralCFRecommender(
    #     "./data/sample_reviews_test.json",
    #     # "./AutoSVDpp/datasets/subsets/yelp_academic_dataset_review.json",
    #     "./NeuralCF/Torch-NCF/checkpoints/checkpoints_neumf_5k_epoch100_l2-0.0000001/pretrain_neumf_factor8neg4_Epoch100_HR0.7258_NDCG0.3565.model",
    #     "5k",
    # )
    # print(model.predict("fr1Hz2acAb3OaL3l6DyKNg"))

    # for r in review_test:
    #     print(model.predict(r["user_id"]))

    svd = SVDSparseCollaborativeFilteringRecommender(user, business, review_train, 5)
    svd.load(open("svd_5913_28028_195455", "r"))
    model = RMSEWeightedEnsembler(
        [
            # AutoSVDPPRecommender(
            #     os.path.abspath(
            #         "./AutoSVDpp/datasets/subsets/restaurant_features_encoded.csv"
            #     )
            # ),
            svd,
            RandomRecommender(business),
        ],
        user,
        business,
        review_train,
    )
    model.fit()

    user_reviews = {u["user_id"]: {} for u in user}
    for r in review_train:
        user_reviews[r["user_id"]][r["business_id"]] = r["stars"]

    for u_id in map(lambda x: x["user_id"], user):
        predicted_vals = model.predict(u_id, top_n=len(business))
        match predicted_vals:
            case Ok(val):
                # we then calculate the rmse
                squared_diffs = [
                    (predicted_star - actual_star) ** 2
                    for predicted_star, actual_star in map(
                        lambda x: (x[2], user_reviews[u_id][x[1]]),
                        filter(lambda x: x[1] in user_reviews[u_id], val),
                    )
                ]
                print(np.sqrt(np.mean(squared_diffs)))
            case Err(msg):
                print(msg)

    # print("svd sparse, k = 5")
    # print("train rmse", model.eval(review_train))
    # print("test rmse", model.eval(review_test))

    # print("random")
    # model = RandomRecommender(business)
    # print("train rmse", model.eval(review_train))
    # print("test rmse", model.eval(review_test))


if __name__ == "__main__":
    main()
