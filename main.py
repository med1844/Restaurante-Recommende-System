from svd_sparse_cf import SVDSparseCollaborativeFilteringRecommender
from data_utils import load_data
from random import choice, seed
from typing import List, Any, Tuple
from random_recommender import RandomRecommender
import json

seed(6220)


def split(data: List[Any], ratio: float) -> Tuple[List[Any], List[Any]]:
    num_element = int(len(data) * ratio)
    return data[:num_element], data[num_element:]


def main():
    user = json.load(open("./data/sample_users.json", "r"))
    business = json.load(open("./data/sample_business.json", "r"))
    review_train = json.load(open("./data/sample_reviews_train.json", "r"))
    review_test = json.load(open("./data/sample_reviews_test.json", "r"))
    model = SVDSparseCollaborativeFilteringRecommender(user, business, review_train, 5)
    model.fit()
    print("svd sparse, k = 5")
    print("train rmse", model.eval(review_train))
    print("test rmse", model.eval(review_test))

    print("random")
    model = RandomRecommender(business)
    print("train rmse", model.eval(review_train))
    print("test rmse", model.eval(review_test))


if __name__ == "__main__":
    main()
