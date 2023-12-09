from typing import List, Optional, Tuple, Dict, Any
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
        self, user_id: str, business_ids: Optional[List[str]] = None, top_n: int = 5
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
