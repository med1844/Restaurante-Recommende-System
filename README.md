# Restaurante Recommende System

## (Optional) Build a dev set

Please refer to the `Make a smaller development set` section in `prototype.ipynb`.

## Migrate your models

You should split the model algo details from methods that converts yelp data into structure that your model could recognize.

Such methods should be implemented in a class that inherits `interface.RestaurantRecommenderInterface`. By defining interface, this allows model aggregators to utilize models in a unified way.

