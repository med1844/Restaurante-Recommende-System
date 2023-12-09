# Restaurante Recommende System

## Models

For model-specific details, please refer to the README.md in each model's folder.

For ensemble model, due to the fact that model folders containing special characters making them impossible to be imported in python, it's stored in `ensemble` branch due to incompatibilities. Once switched to `ensemble` branch, run `python main.py` to run the ensemble model.

## (Optional) Build a dev set

```python
from data_utils import load_data, save_data, gen_subset

in_dataset_root = "/mnt/d/Download/yelp/"
out_dataset_root = "/mnt/d/Download/yelp/subset/"
filenames = ("yelp_academic_dataset_user.json", "yelp_academic_dataset_business.json", "yelp_academic_dataset_review.json")

user, business, review = load_data(in_dataset_root, filenames)
s_user, s_business, s_review = gen_subset(user, business, review, 100, 100, 1000)  # generate a subset with 100 user, 100 business and 1000 reviews
save_data((s_user, s_business, s_review), out_dataset_root, filenames)
```

Change `dataset_root` to where you put your dataset. You might also want to change `filenames`.

## Develop your models

The `master` branch should only contain code / files that has no dependency on models, such as:
0. Model interfaces (to read trained models)
1. Data utilities (to read data & provide human-readable prediction result)
2. Application code, e.g.
    - Flask app that would be deployed on AWS EC2
    - A handler function that would be deployed on AWS Lambda
3. Front-end files

To train & test your model, please create a new branch. They will be merged into `master` branch after review.

## Migrate your models

You should split the model algo details from methods that converts yelp data into structure that your model could recognize.

Such methods should be implemented in a class that inherits `interface.RestaurantRecommenderInterface`. By defining interface, this allows model aggregators to utilize models in a unified way.

### Serialization & Deserialization

The application would be deployed on server with limited hardware capabilities, i.e. no training, only inference. Thus you must implement `interface.Serializable` and `interface.Deserializable` to save & load model.

Please write tests for your model to make sure that after `save` and `load`, the model data is not changed.

For more info on testing model data, you may refer to function `test_save_n_load_identity` in `tests/test_svd_sparse_cf.py`.

