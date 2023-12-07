from typing import List, Tuple
from interface import RestaurantRecommenderInterface, Json
from random import randint
from result import Result, Ok, Err
import numpy as np


class RandomRecommender(RestaurantRecommenderInterface):
    def __init__(self, business: List[Json]) -> None:
        self.business = business
        # self.business_id_map = {
        #     business["business_id"]: i for i, business in enumerate(business)
        # }

    def fit(self):
        pass

    def predict(
        self, user_id: str, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        raise NotImplementedError

    def eval(self, review: List[Json]) -> float:
        predicted_vals = []
        for r in review:
            predicted_vals.append(randint(1, 5))
        # we then calculate the rmse
        squared_diffs = [
            (predicted_star - actual_star) ** 2
            for predicted_star, actual_star in zip(
                predicted_vals, [r["stars"] for r in review]
            )
        ]
        return np.sqrt(np.mean(squared_diffs))
