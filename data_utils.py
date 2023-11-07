from typing import Tuple, List, Any
import os
import json


def load_data(
    data_folder: str,
    filenames=("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json")
) -> Tuple[List[Any], ...]:
    return tuple(map(
        lambda f: list(map(json.loads, open(os.path.join(data_folder, f), "r", encoding="utf-8").readlines())),
        filenames
    ))


def save_data(
    data: Tuple[List[Any], ...],
    data_folder: str,
    filenames=("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json"),
):
    for d, f in zip(data, filenames):
        open(os.path.join(data_folder, f), "w").write("\n".join(map(json.dumps, d)))
