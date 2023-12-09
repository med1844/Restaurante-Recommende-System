from typing import Tuple, List, Any
import os
import json
from utils.interface import Json
from collections import Counter
from random import sample


def load_data(
    data_folder: str,
    filenames=(
        "yelp_academic_dataset_user.json",
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_review.json",
    ),
    line=True,
) -> Tuple[List[Any], ...]:
    match line:
        case True:
            return tuple(
                map(
                    lambda f: list(
                        map(
                            json.loads,
                            open(
                                os.path.join(data_folder, f), "r", encoding="utf-8"
                            ).readlines(),
                        )
                    ),
                    filenames,
                )
            )
        case False:
            return tuple(
                map(
                    lambda f: json.load(open(os.path.join(data_folder, f), "r")),
                    filenames,
                )
            )


def save_data(
    data: Tuple[List[Any], ...],
    data_folder: str,
    filenames=(
        "yelp_academic_dataset_user.json",
        "yelp_academic_dataset_business.json",
        "yelp_academic_dataset_review.json",
    ),
):
    for d, f in zip(data, filenames):
        open(os.path.join(data_folder, f), "w").write("\n".join(map(json.dumps, d)))


def gen_subset(
    user_data: List[Json],
    business_data: List[Json],
    review_data: List[Json],
    user_n: int,
    business_n: int,
    review_n: int,
) -> Tuple[List[Json], List[Json], List[Json]]:
    # To control sparsity of the subset data, the algorithm is designed in a way differs from pure random sampling:
    # 0. select top `user_n` users that posts the most reviews
    # 1. among these reviews, select top `business_n` businesses that receives the most reviews
    # 2. among all reviews, select reviews posted by both top `user_n` users and to top `business_n` businesses
    # 3. randomly sample min(review_n, len(result of step 2)) reviews to form review subset
    user_cnt = Counter(review["user_id"] for review in review_data)
    user_cnt = list(user_cnt.items())
    user_cnt.sort(key=lambda x: x[1], reverse=True)

    subset_user_id = {user_id for user_id, _ in user_cnt[:user_n]}
    subset_user_data = [user for user in user_data if user["user_id"] in subset_user_id]
    subset_review_data = [
        review for review in review_data if review["user_id"] in subset_user_id
    ]

    subset_business_id_cnt = Counter(
        [review["business_id"] for review in subset_review_data]
    )
    subset_business_id_cnt = list(subset_business_id_cnt.items())
    subset_business_id_cnt.sort(key=lambda x: x[1], reverse=True)

    subset_business_id = {
        business_id for business_id, _ in subset_business_id_cnt[:business_n]
    }
    subset_business_data = [
        business
        for business in business_data
        if business["business_id"] in subset_business_id
    ]

    subset_review_data = [
        review
        for review in review_data
        if review["user_id"] in subset_user_id
        and review["business_id"] in subset_business_id
    ]
    subset_review_data = sample(
        subset_review_data, min(review_n, len(subset_review_data))
    )

    return subset_user_data, subset_business_data, subset_review_data
