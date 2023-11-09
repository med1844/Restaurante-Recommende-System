from io import TextIOWrapper
import json
from interface import RestaurantRecommenderInterface, Json, Serializable, Deserializable
from typing import List, Optional, Tuple
import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix, lil_matrix
from result import Result, Ok, Err
from heapq import heappush, heappop
import pickle


class SVDSparseCollaborativeFilteringModel(Serializable, Deserializable):
    def __init__(self, k: int) -> None:
        self.u: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.vt: Optional[np.ndarray] = None
        self.k = k

    def fit(self, x: List[Tuple[int, int]], y: List[float]):
        matrix = lil_matrix((max(map(lambda t: t[0], x)) + 1, max(map(lambda t: t[1], x)) + 1), dtype=float)
        for (i, j), val in zip(x, y):
            matrix[i, j] = val
        u, s, vt = svds(matrix, k=self.k)
        self.u = u
        self.s = s
        self.vt = vt

    def predict(self, x: List[Tuple[int, int]], top_n: int = 5) -> Result[List[Tuple[Tuple[int, int], float]], str]:
        # by default return top n
        if self.u is not None and self.s is not None and self.vt is not None:
            heap = []
            for i, j in x:
                heappush(heap, (float(self.u[i, :].dot(np.diag(self.s).dot(self.vt[:, j]))), (i, j)))
                if len(heap) > top_n:
                    heappop(heap)
            return Ok([((i, j), val) for val, (i, j) in sorted(heap, key=lambda x: x[1], reverse=True)])
        return Err("not fit yet")

    def save(self, fp: TextIOWrapper) -> Result[None, str]:
        if self.u is not None and self.s is not None and self.vt is not None:
            json.dump({
                "u": json.dumps(pickle.dumps(self.u).decode("latin-1")),
                "s": json.dumps(pickle.dumps(self.s).decode("latin-1")),
                "vt": json.dumps(pickle.dumps(self.vt).decode("latin-1")),
            }, fp)
            return Ok(None)
        return Err("not fit yet")

    def load(self, fp: TextIOWrapper):
        obj = json.load(fp)
        self.u = pickle.loads(json.loads(obj["u"]).encode("latin-1"))
        self.s = pickle.loads(json.loads(obj["s"]).encode("latin-1"))
        self.vt = pickle.loads(json.loads(obj["vt"]).encode("latin-1"))


class SVDSparseCollaborativeFilteringRecommender(RestaurantRecommenderInterface, Serializable, Deserializable):
    def __init__(self, user: List[Json], business: List[Json], review: List[Json], k: int) -> None:
        self.model = SVDSparseCollaborativeFilteringModel(k)
        self.user = user
        self.business = business
        self.review = review
        self.user_id_map = {user["user_id"]: i for i, user in enumerate(user)}
        self.business_id_map = {business["business_id"]: i for i, business in enumerate(business)}

    def fit(self):
        self.model.fit([(self.user_id_map[r["user_id"]], self.business_id_map[r["business_id"]]) for r in self.review], [r["stars"] for r in self.review])

    def predict(self, user_id: str, top_n: int = 5) -> Result[List[Tuple[str, str, float]], str]:
        match self.model.predict([(self.user_id_map[user_id], i) for i in self.business_id_map.values()], top_n):
            case Ok(res):
                return Ok([(self.business[i]["name"], self.business[i]["business_id"], val) for (_, i), val in res])
            case Err(msg):
                return Err(msg)

    def save(self, fp: TextIOWrapper) -> Result[None, str]:
        return self.model.save(fp)

    def load(self, fp: TextIOWrapper):
        self.model.load(fp)
