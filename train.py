from svd_sparse_cf import SVDSparseCollaborativeFilteringRecommender
from data_utils import load_data
import os


user_subset, business_subset, review_subset = load_data(
    os.path.abspath("data"),
    (
        "sample_users.json",
        "sample_business.json",
        "sample_reviews_train.json",
    ),
    line=False,
)


r = SVDSparseCollaborativeFilteringRecommender(
    user_subset, business_subset, review_subset, 5
)

r.fit()

r.save(
    open(
        "svd_%d_%d_%d" % (len(user_subset), len(business_subset), len(review_subset)),
        "w",
    )
)
