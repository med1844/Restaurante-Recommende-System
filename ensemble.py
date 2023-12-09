from interface import RestaurantRecommenderInterface, Json
from typing import Iterable, List, Optional, Set, Tuple, Dict
from result import Result, Ok, Err
import math
import numpy as np
import tqdm


# def dcg(scores: List[int]) -> float:
#     return sum((2**rel - 1) / math.log2(i + 2) for i, rel in enumerate(scores))


# def ndcg(predicted: List[int], ground_truth: List[int]) -> float:
#     dcg_pred = dcg([ground_truth[i] for i in predicted])

#     sorted_truth = sorted(ground_truth, reverse=True)
#     idcg = dcg(sorted_truth)

#     return dcg_pred / idcg if idcg else 0


# def assign_weights(ndcg_scores: List[float]) -> List[float]:
#     """Assign weights based on NDCG scores."""
#     total_score = sum(ndcg_scores)
#     # Normalize scores to sum up to 1
#     weights = (
#         [score / total_score for score in ndcg_scores]
#         if total_score
#         else [0.0] * len(ndcg_scores)
#     )
#     return weights


def calc_rmse(predicted_vals: Iterable[float], actual_vals: Iterable[float]) -> float:
    squared_diffs = [
        (predicted_star - actual_star) ** 2
        for predicted_star, actual_star in zip(predicted_vals, actual_vals)
    ]
    return np.sqrt(np.mean((np.array(predicted_vals) - np.array(actual_vals)) ** 2))


class RMSEWeightedEnsembler(RestaurantRecommenderInterface):
    def __init__(
        self,
        models: List[RestaurantRecommenderInterface],
        users: List[Json],
        businesses: List[Json],
        reviews_train: List[Json],
    ) -> None:
        self.models = models
        self.model_weight = [0.0] * len(self.models)
        self.users = users
        self.businesses = businesses
        self.reviews_train = reviews_train

    def fit(self):
        for model in self.models:
            model.fit()
        # calculate ndcg on training set, evaluate weight
        # simply rank everything, then extract the ranking of ground truth
        for user in tqdm.tqdm(self.users[:5]):
            user_reviews: Dict[str, int] = {
                review["business_id"]: review["stars"]
                for review in self.reviews_train
                if review["user_id"] == user["user_id"]
            }
            model_reciprocal_rmses = []
            for model in self.models:
                pred_rating = model.predict(
                    user["user_id"],
                    list(user_reviews.keys()),
                    top_n=len(user_reviews),
                )
                match pred_rating:
                    case Ok(value):
                        pred_rating = [
                            rating
                            for _, business_id, rating in value
                            if business_id in user_reviews
                        ]
                        gt_rating = [
                            user_reviews[business_id]
                            for _, business_id, _ in value
                            if business_id in user_reviews
                        ]
                        rmse = calc_rmse(pred_rating, gt_rating)
                        model_reciprocal_rmses.append(1 / rmse)
                    case Err(value):
                        raise ValueError(
                            "model %r goes wrong when predict in ensembler fit, reason: %s"
                            % (model, value)
                        )
            rmse_sum = sum(model_reciprocal_rmses)
            for i, val in enumerate(model_reciprocal_rmses):
                self.model_weight[i] += val / rmse_sum

        acc_rmse_sum = sum(self.model_weight)
        for i in range(0, len(self.model_weight)):
            self.model_weight[i] /= acc_rmse_sum

    def predict(
        self, user_id: str, business_ids: Optional[List[str]] = None, top_n: int = 5
    ) -> Result[List[Tuple[str, str, float]], str]:
        # each predicted business rating multiplies with the model weight
        # if there's more models recommending the same one, that one got add up score
        business_id_accu_rating = {}
        for model, model_w in zip(self.models, self.model_weight):
            match model.predict(user_id, business_ids, top_n):
                case Ok(recommendation):
                    for b_name, b_id, rating in recommendation:
                        business_id_accu_rating.setdefault((b_name, b_id), 0.0)
                        business_id_accu_rating[(b_name, b_id)] += rating * model_w
                case Err(msg):
                    return Err(msg)
        top_ratings = list(business_id_accu_rating.items())
        top_ratings.sort(
            key=lambda tup: tup[1], reverse=True
        )  # stupid python, should be key=|(_, pred)| pred
        return Ok([(b_name, b_id, r) for (b_name, b_id), r in top_ratings[:top_n]])
