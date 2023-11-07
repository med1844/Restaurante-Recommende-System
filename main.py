from svd_sparse_cf import SVDSparseCollaborativeFilteringWrapperModel
from data_utils import load_data
from random import choice


def main():
    user, business, review = load_data("/mnt/d/Download/yelp/subset/", ("subset_user.json", "subset_business.json", "subset_review.json"))
    model = SVDSparseCollaborativeFilteringWrapperModel(user, business, review, 5)
    model.fit()
    print(model.predict(choice(user)["user_id"]))


if __name__ == "__main__":
    main()
