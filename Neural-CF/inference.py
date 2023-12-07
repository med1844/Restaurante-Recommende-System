import torch
import pandas as pd
import numpy as np
from .neumf import NeuMFArchitecture
from .data import YelpLoader, SampleGenerator
from collections import OrderedDict
import argparse

class RecommenderSystem:
    def __init__(self, data_dir, model_checkpoint, user_id, topK):
        self.data_dir = data_dir
        self.model_checkpoint = model_checkpoint
        self.user_id = user_id
        self.converted_userId = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.topK = topK

    def load_data(self):
        yelp_loader = YelpLoader(self.data_dir)
        yelp_rating = yelp_loader.get_yelp_rating()   

        sample_generator = SampleGenerator(ratings=yelp_rating)
        negative = sample_generator._sample_negative(ratings=yelp_rating)

        predict_data = pd.merge(yelp_rating, negative[['userId','negative_samples']], on='userId')

        _users, _items, negative_users, negative_items = [], [], [], []
        for row in predict_data.itertuples():
            _users.append(int(row.userId))
            _items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))

        return yelp_rating, [torch.LongTensor(_users), torch.LongTensor(_items),
                torch.LongTensor(negative_users), torch.LongTensor(negative_items)]

    def convert_userID(self, rating):
        """ Convert the dataset userId with user idx ID """

        userId_conversion = rating.groupby('userId')['user_id'].unique().reset_index()
        desired_user_name = self.user_id[1:-1]
        is_present = desired_user_name in userId_conversion['user_id'].values
        assert is_present, f"NCF Model Fail to Predict. User {desired_user_name} is not present"
        self.converted_userId = userId_conversion[userId_conversion['user_id'] == desired_user_name]['userId'].values[0]

    def convert_businessID(self, rating, itemId):
        """ Convert the dataset business ID with preprocessed IDs """

        businessId_conversion = rating.groupby('itemId')['business_id'].unique().reset_index()
        is_present = itemId in businessId_conversion['itemId'].values
        if is_present:
          converted_businessId = businessId_conversion[businessId_conversion['itemId'] == itemId]['business_id'].values[0]
          return converted_businessId    
    
    def load_model(self):
        neumf = torch.load(self.model_checkpoint)
        config = {
            'num_users': 961,
            'num_items': 1000,
            'latent_dim_mf': 8,
            'latent_dim_mlp': 8,
            'num_negative': 4,
            'layers': [16, 64, 32, 16, 8],
            'use_cuda': True,
        }

        neumf_model = NeuMFArchitecture(config)
        neumf_model.load_state_dict(neumf)
        neumf_model = neumf_model.to(self.device)
        neumf_model.eval()

        return neumf_model

    def infer(self, rating, predict_data):
        self.convert_userID(rating)

        with torch.no_grad():
            test_users, test_items = predict_data[0].to(self.device), predict_data[1].to(self.device)
            negative_users, negative_items = predict_data[2].to(self.device), predict_data[3].to(self.device)

            test_scores = self.neumf_model(test_users, test_items)
            negative_scores = self.neumf_model(negative_users, negative_items)

        _users = negative_users
        _items = negative_items
        _scores = negative_scores

        user_array = _users.cpu().numpy()
        item_array = _items.cpu().numpy()
        scores_array = _scores.cpu().numpy().flatten()

        unique_user_ids = np.unique(user_array)

        reordered_scores = np.zeros_like(scores_array)
        reordered_items = np.zeros_like(item_array)

        for user_id in unique_user_ids:
            if user_id == self.converted_userId:
                user_mask = (user_array == user_id)
                user_scores = scores_array[user_mask]
                items = item_array[user_mask]

                ordered_indices = np.argsort(user_scores)[::-1]
                reordered_scores[user_mask] = user_scores[ordered_indices]
                reordered_items[user_mask] = items[ordered_indices]

                unique_reordered_items = list(OrderedDict.fromkeys(reordered_items[user_mask]))
                topK_item = unique_reordered_items[:self.topK]
                return topK_item

def parse_args():
    parser = argparse.ArgumentParser(description="Recommender System")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to the Yelp review.json data directory")
    parser.add_argument("--model_checkpoint", type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument("--userId", type=str, required=True, help="User ID from Yelp Dataset")
    parser.add_argument("--topK", type=int, default=10, help="Number of top recommendation you want to display")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    print('-'*10 + f'Top {args.topK} for {args.userId}' + '-'*10)
    recommender = RecommenderSystem(args.data_dir, args.model_checkpoint, args.userId, args.topK)
    rating, predict_data = recommender.load_data()
    recommender.neumf_model = recommender.load_model()

    topK_item = recommender.infer(rating, predict_data)
    topK_item_converted = []
    for item in topK_item:
      businessId = recommender.convert_businessID(rating=rating, itemId = item)
      topK_item_converted.append(businessId)

    top_item_recommended = np.concatenate(topK_item_converted, axis=0)
    print('')
    print(top_item_recommended)
