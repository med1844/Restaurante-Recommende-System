from utils.LoadDataSet import LoadData
from models.AutoSVDpp import *

# train_data, test_data = LoadData().loadSmallDataSet()
train_data, test_data = LoadData().loadSampleDataSet()
# autosvdpp = AutoSVDpp(path_of_feature_file="datasets/smallsets/restaurant_features_encoded.csv")
autosvdpp = AutoSVDpp(path_of_feature_file="datasets/samplesets/restaurant_features_encoded.csv")
autosvdpp.train(train_data=train_data, test_data=test_data)
# autosvdpp.evaluate(test_data)
autosvdpp.save_model()
