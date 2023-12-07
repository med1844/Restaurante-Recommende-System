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
    def __init__(self, k: int, num_user: int, num_business: int) -> None:
        self.u: Optional[np.ndarray] = None
        self.s: Optional[np.ndarray] = None
        self.vt: Optional[np.ndarray] = None
        self.k = k
        self.num_user = num_user
        self.num_business = num_business

    def fit(self, x: List[Tuple[int, int]], y: List[float]):
        matrix = lil_matrix(
            (self.num_user, self.num_business),
            dtype=float,
        )
        for (i, j), val in zip(x, y):
            matrix[i, j] = val
        u, s, vt = svds(matrix, k=self.k)
        self.u = u
        self.s = s
        self.vt = vt

    def predict(
        self, x: List[Tuple[int, int]], top_n: int = 5
    ) -> Result[List[Tuple[Tuple[int, int], float]], str]:
        # by default return top n
        if self.u is not None and self.s is not None and self.vt is not None:
            heap = []
            for i, j in x:
                heappush(
                    heap,
                    (
                        float(self.u[i, :].dot(np.diag(self.s).dot(self.vt[:, j]))),
                        (i, j),
                    ),
                )
                if len(heap) > top_n:
                    heappop(heap)
            return Ok(
                [
                    ((i, j), val)
                    for val, (i, j) in sorted(heap, key=lambda x: x[1], reverse=True)
                ]
            )
        return Err("not fit yet")

    def save(self, fp: TextIOWrapper) -> Result[None, str]:
        if self.u is not None and self.s is not None and self.vt is not None:
            json.dump(
                {
                    "u": json.dumps(pickle.dumps(self.u).decode("latin-1")),
                    "s": json.dumps(pickle.dumps(self.s).decode("latin-1")),
                    "vt": json.dumps(pickle.dumps(self.vt).decode("latin-1")),
                },
                fp,
            )
            return Ok(None)
        return Err("not fit yet")

    def load(self, fp: TextIOWrapper):
        obj = json.load(fp)
        self.u = pickle.loads(json.loads(obj["u"]).encode("latin-1"))
        self.s = pickle.loads(json.loads(obj["s"]).encode("latin-1"))
        self.vt = pickle.loads(json.loads(obj["vt"]).encode("latin-1"))


class SVDSparseCollaborativeFilteringRecommender(
    RestaurantRecommenderInterface, Serializable, Deserializable
):
    def __init__(
        self, user: List[Json], business: List[Json], review: List[Json], k: int
    ) -> None:
        self.model = SVDSparseCollaborativeFilteringModel(k, len(user), len(business))
        self.user = user
        self.business = business
        self.review = review
        self.user_id_map = {user["user_id"]: i for i, user in enumerate(user)}
        self.business_id_map = {
            business["business_id"]: i for i, business in enumerate(business)
        }
        self.user_star_sum = [0] * len(self.user)
        self.user_star_cnt = [0] * len(self.user)
        for r in self.review:
            self.user_star_sum[self.user_id_map[r["user_id"]]] += r["stars"]
            self.user_star_cnt[self.user_id_map[r["user_id"]]] += 1
        self.user_star_avg = [
            s / c for s, c in zip(self.user_star_sum, self.user_star_cnt)
        ]
        self.user_min = [5] * len(self.user)
        self.user_max = [1] * len(self.user)
        for r in self.review:
            i = self.user_id_map[r["user_id"]]
            star = r["stars"] - self.user_star_avg[i]
            self.user_min[i] = min(self.user_min[i], star)
            self.user_max[i] = max(self.user_max[i], star)

    @staticmethod
    def into_unit(v: int, min_: int, max_: int) -> float:
        return (v - min_) / (max_ - min_)

    @staticmethod
    def from_unit(v: float, min_: int, max_: int) -> float:
        return v * (max_ - min_) + min_

    def fit(self):
        self.model.fit(
            [
                (self.user_id_map[r["user_id"]], self.business_id_map[r["business_id"]])
                for r in self.review
            ],
            [
                self.into_unit(
                    s - self.user_star_avg[i],
                    self.user_min[i],
                    self.user_max[i],
                )
                for i, s in map(
                    lambda r: (self.user_id_map[r["user_id"]], r["stars"]), self.review
                )
            ],
        )

    def predict(
        self, user_id: str, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        user_i = self.user_id_map[user_id]
        user_bias = self.user_star_avg[user_i]
        match self.model.predict(
            [(self.user_id_map[user_id], i) for i in self.business_id_map.values()],
            top_n,
        ):
            case Ok(res):
                return Ok(
                    [
                        (
                            self.business[i]["name"],
                            self.business[i]["business_id"],
                            self.from_unit(
                                val, self.user_min[user_i], self.user_max[user_i]
                            )
                            + user_bias,
                        )
                        for (_, i), val in res
                    ]
                )
            case Err(msg):
                return Err(msg)

    def eval(self, review: List[Json]) -> float:
        predicted_vals = []
        for r in review:
            match self.model.predict(
                [
                    (
                        self.user_id_map[r["user_id"]],
                        self.business_id_map[r["business_id"]],
                    )
                ],
                1,
            ):
                case Ok(res):
                    predicted_vals.append(min(5.0, max(1.0, res[0][1])))
                case Err(msg):
                    raise Exception(msg)
        # we then calculate the rmse
        squared_diffs = [
            (predicted_star - actual_star) ** 2
            for predicted_star, actual_star in zip(
                predicted_vals, [r["stars"] for r in review]
            )
        ]
        return np.sqrt(np.mean(squared_diffs))

    def save(self, fp: TextIOWrapper) -> Result[None, str]:
        return self.model.save(fp)

    def load(self, fp: TextIOWrapper):
        self.model.load(fp)
