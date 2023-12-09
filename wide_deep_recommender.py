from typing import List, Tuple, Optional
from interface import RestaurantRecommenderInterface, Json
import pandas as pd
from result import Result, Ok


class WideDeepRecommender(RestaurantRecommenderInterface):
    def __init__(self, business: List[Json]) -> None:
        self.df = pd.read_pickle("rating_df.pkl")
        self.business_id_to_name = {b["business_id"]: b["name"] for b in business}

    def fit(self):
        pass

    def predict(
        self, user_id: str, business_ids: Optional[List[str]] = None, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        if business_ids is None:
            business_ids = list(self.df.keys())
        return Ok(
            [
                (self.business_id_to_name[b_id], b_id, self.df.loc[user_id, b_id])
                for b_id in business_ids
            ]
        )
