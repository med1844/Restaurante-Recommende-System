import argparse
from typing import Optional, Literal
import torch
import pandas as pd
import numpy as np
from neumf import NeuMFArchitecture
from data import YelpLoader, SampleGenerator
from tqdm import tqdm

from collections import OrderedDict


class RecommenderSystem:
    def __init__(self, data_dir, model_checkpoint, model_size: Literal["1k", "5k"]):
        self.data_dir = data_dir
        self.model_checkpoint = model_checkpoint
        self.converted_userId = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_size = model_size
        self.neumf_model: Optional[NeuMFArchitecture] = None

        neumf = torch.load(self.model_checkpoint, map_location=self.device)
        if self.model_size == "1k":
            config = {
                "num_users": 961,
                "num_items": 1000,
                "latent_dim_mf": 8,
                "latent_dim_mlp": 8,
                "num_negative": 4,
                "layers": [16, 64, 32, 16, 8],
                "use_cuda": True,
            }
        elif self.model_size == "5k":
            config = {
                "num_users": 5000,
                "num_items": 5000,
                "latent_dim_mf": 8,
                "latent_dim_mlp": 8,
                "num_negative": 4,
                "layers": [16, 64, 32, 16, 8],
                "use_cuda": True,
            }
        else:
            raise ValueError("invalid model size, must be 1k or 5k")

        neumf_model = NeuMFArchitecture(config)
        neumf_model.load_state_dict(neumf)
        neumf_model = neumf_model.to(self.device)
        neumf_model.eval()

        self.neumf_model = neumf_model

    def load_data(self):
        yelp_loader = YelpLoader(self.data_dir)
        yelp_rating = yelp_loader.get_yelp_rating()

        sample_generator = SampleGenerator(ratings=yelp_rating)
        negative = sample_generator._sample_negative(ratings=yelp_rating)

        predict_data = pd.merge(
            yelp_rating, negative[["userId", "negative_samples"]], on="userId"
        )

        _users, _items, negative_users, negative_items = [], [], [], []
        for row in predict_data.itertuples():
            _users.append(int(row.userId))
            _items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))

        print("Finish Loading Data...")
        return yelp_rating, [
            torch.LongTensor(_users),
            torch.LongTensor(_items),
            torch.LongTensor(negative_users),
            torch.LongTensor(negative_items),
        ]

    def convert_userID(self, user_id: str, rating):
        """Convert the dataset userId with user idx ID"""

        userId_conversion = rating.groupby("userId")["user_id"].unique().reset_index()
        # desired_user_name = user_id[1:-1]
        is_present = user_id in userId_conversion["user_id"].str[0].values
        assert is_present, f"NCF Model Fail to Predict. User {user_id} is not present"
        self.converted_userId = userId_conversion[
            userId_conversion["user_id"] == user_id
        ]["userId"].values[0]
        return self.converted_userId

    def convert_businessID(self, rating, itemId):
        """Convert the dataset business ID with preprocessed IDs"""

        businessId_conversion = (
            rating.groupby("itemId")["business_id"].unique().reset_index()
        )
        is_present = itemId in businessId_conversion["itemId"].values
        if is_present:
            converted_businessId = businessId_conversion[
                businessId_conversion["itemId"] == itemId
            ]["business_id"].values[0]
            return converted_businessId

    # def load_model(self):

    def infer(self, user_id, rating, predict_data):
        self.convert_userID(user_id, rating)

        with torch.no_grad():
            test_users, test_items = (
                predict_data[0].to(self.device),
                predict_data[1].to(self.device),
            )
            negative_users, negative_items = (
                predict_data[2].to(self.device),
                predict_data[3].to(self.device),
            )

            test_scores = self.neumf_model(test_users, test_items)
            negative_scores = self.neumf_model(negative_users, negative_items)

        _users = negative_users
        _items = negative_items
        _scores = negative_scores

        user_array = _users.cpu().numpy()
        item_array = _items.cpu().numpy()
        scores_array = _scores.cpu().numpy().flatten()
        return user_array, item_array, scores_array

    def predict(self, rating, predict_data, target_user_id, top_k: int):
        user_array, item_array, scores_array = self.infer(
            target_user_id, rating, predict_data
        )
        unique_user_indexs = np.unique(user_array)

        reordered_scores = np.zeros_like(scores_array)
        reordered_items = np.zeros_like(item_array)

        target_user_index = self.convert_userID(target_user_id, rating)

        for user_index in unique_user_indexs:
            if user_index == target_user_index:
                user_mask = user_array == user_index
                user_scores = scores_array[user_mask]
                items = item_array[user_mask]

                ordered_indices = np.argsort(user_scores)[::-1]
                reordered_scores[user_mask] = user_scores[ordered_indices]
                reordered_items[user_mask] = items[ordered_indices]

                unique_reordered_items = list(
                    OrderedDict.fromkeys(reordered_items[user_mask])
                )
                topK_item = unique_reordered_items[:top_k]
                return topK_item

    def eval4sample(self, rating, predict_data, top_k: int):
        """
        Evaluate the model by runing the sample test subdataset created by the group
        """

        # Load in sample test data
        sample_dir = "/content/drive/MyDrive/Neural-CF/Yelp-Dataset/subset/sample_reviews_test_reformat.json"
        sample_test = pd.read_json(sample_dir, lines=True)

        # Keep only data which the stars is greater than 4
        sample_test = sample_test[sample_test["stars"] >= 3]
        unique_user_id = sample_test["user_id"].unique().tolist()
        userId_conversion = rating.groupby("userId")["user_id"].unique().reset_index()

        # Group by 'user_id' and aggregate 'business_id' into lists, removing duplicates
        user_business_dict = (
            sample_test.groupby("user_id")["business_id"]
            .agg(lambda x: list(set(x)))
            .to_dict()
        )

        # Run Inference
        user_array, item_array, scores_array = self.infer(rating, predict_data)
        unique_user_ids = np.unique(user_array)
        reordered_scores = np.zeros_like(scores_array)
        reordered_items = np.zeros_like(item_array)

        user_not_found = []
        hit = 0
        count = 0
        for desired_user_name in tqdm(
            unique_user_id, desc="Processing Users", unit="user"
        ):
            # if count == 30:
            #     break

            is_present = desired_user_name in userId_conversion["user_id"].values
            if is_present:
                count += 1  # For debugging

                # Encode the dataset userID
                converted_userId = userId_conversion[
                    userId_conversion["user_id"] == desired_user_name
                ]["userId"].values[0]
                # print(f'Processing userID: {desired_user_name}, {converted_userId}')
                groundtruth_item = user_business_dict[desired_user_name]

                # Find TopK Prediction
                for user_id in unique_user_ids:
                    if user_id == converted_userId:
                        user_mask = user_array == user_id
                        user_scores = scores_array[user_mask]
                        items = item_array[user_mask]

                        ordered_indices = np.argsort(user_scores)[::-1]
                        reordered_scores[user_mask] = user_scores[ordered_indices]
                        reordered_items[user_mask] = items[ordered_indices]

                        unique_reordered_items = list(
                            OrderedDict.fromkeys(reordered_items[user_mask])
                        )
                        topK_item = unique_reordered_items[top_k]
                        print(topK_item)

                        # Convert encoded item back to dataset businessID
                        topK_item_converted = []
                        for item in topK_item:
                            businessId = recommender.convert_businessID(
                                rating=rating, itemId=item
                            )
                            topK_item_converted.append(businessId)
                        top_item_recommended = np.concatenate(
                            topK_item_converted, axis=0
                        )

                        # Compare predicted item with Ground truth item
                        gt = set(groundtruth_item)
                        pred = set(top_item_recommended)
                        if bool(gt.intersection(pred)):
                            hit += 1

            # Missed User
            else:
                user_not_found.append(desired_user_name)

        print(f"Total Missed user number: {len(user_not_found)}")
        print(f"Total Tested User: {count}")
        print(f"Total number of true prediction: {hit}")


def parse_args():
    parser = argparse.ArgumentParser(description="Recommender System")
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to the Yelp review.json data directory",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--userId", type=str, required=True, help="User ID from Yelp Dataset"
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=10,
        help="Number of top recommendation you want to display",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="1k",
        help="Choose between model trained on 5k user or 1k user",
    )
    parser.add_argument(
        "--run_eval", action="store_true", help="Run evaluation on the sample test set"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    recommender = RecommenderSystem(
        args.data_dir, args.model_checkpoint, args.userId, args.topK, args.model
    )
    rating, predict_data = recommender.load_data()
    recommender.neumf_model = recommender.load_model()

    if not args.run_eval:
        print("-" * 10 + f"Top {args.topK} for {args.userId}" + "-" * 10)
        topK_item = recommender.predict(rating, predict_data, args.topK)
        topK_item_converted = []
        for item in topK_item:
            businessId = recommender.convert_businessID(rating=rating, itemId=item)
            topK_item_converted.append(businessId)

        top_item_recommended = np.concatenate(topK_item_converted, axis=0)
        print("")
        print(top_item_recommended)

    else:
        print(
            "-" * 10 + f"Running Evaluation with Top {args.topK} Prediction" + "-" * 10
        )
        recommender.eval4sample(rating, predict_data)
