from typing import Literal, List, Tuple
from interface import RestaurantRecommenderInterface
from .inference import RecommenderSystem
import os
from functools import partial
from result import Result, Ok


class NeuralCFRecommender(RestaurantRecommenderInterface):
    def __init__(
        self, review_json: str, model_checkpoint: str, model_size: Literal["1k", "5k"]
    ) -> None:
        self.model = RecommenderSystem(review_json, model_checkpoint, model_size)
        self.rating, self.predict_data = self.model.load_data()

    def fit(self):
        pass

    def predict(
        self, user_id: str, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        top_k_item = self.model.predict(
            self.rating,
            self.predict_data,
            user_id,
            top_n,
        )
        return Ok(top_k_item)
