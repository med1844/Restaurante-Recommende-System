from typing import List, Tuple, Dict, Any
from abc import ABC, abstractmethod
from result import Result
from io import TextIOWrapper


Json = Dict[str, Any]


class RestaurantRecommenderInterface(ABC):
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(
        self, user_id: str, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        # return a list of (business name, business_id, estimated rating)
        pass


class Serializable(ABC):
    @abstractmethod
    def save(self, fp: TextIOWrapper) -> Result[None, str]:
        pass


class Deserializable(ABC):
    @abstractmethod
    def load(self, fp: TextIOWrapper):
        pass
