from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from result import Result


Json = Dict[str, Any]


class RestaurantRecommenderInterface(ABC):
    @abstractmethod
    def __init__(self, user: List[Json], business: List[Json], review: List[Json]) -> None:
        super().__init__()

    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self, user_id: str, top_n: int = 5) -> Result[List[Tuple[str, str, float]], str]:
        # return a list of (business name, business_id, estimated rating)
        pass


class Serializable(ABC):
    @abstractmethod
    def save(self, filename: str):
        pass


class Deserializable(ABC):
    @abstractmethod
    @classmethod
    def load(cls, filename: str):
        pass

