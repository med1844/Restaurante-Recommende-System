from utils.LoadDataSet import LoadData
from models.AutoSVD import AutoSVD
import numpy as np
import os

train_data, test_data = LoadData().loadSmallDataSet()
autosvd = AutoSVD(path_of_feature_file="datasets/subsets/restaurant_features_encoded.csv")
autosvd.train(train_data=train_data, test_data=test_data)
# autosvd.evaluate(test_data)
autosvd.save_model()