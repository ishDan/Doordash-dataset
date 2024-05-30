# Delivery Duration Prediction

This project aims to predict the delivery duration of orders in a food delivery system. The dataset used for this analysis includes historical data of food delivery orders. The dataset contains various features related to each order, such as order details, store information, delivery time, and more.

## Problem Statement

The task is to build a predictive model that accurately estimates the delivery duration of an order based on various factors such as order details, store information, time of the day, and other relevant features. Accurate prediction of delivery duration is crucial for optimizing delivery operations, improving customer satisfaction, and managing resources efficiently.

## Dataset Description

The dataset consists of the following main features:

- `created_at`: Timestamp indicating the time when the order was created.
- `actual_delivery_time`: Timestamp indicating the actual delivery time of the order.
- `total_items`: Total number of items in the order.
- `subtotal`: Subtotal amount of the order.
- `num_distinct_items`: Number of distinct items in the order.
- `min_item_price`: Minimum price of an item in the order.
- `max_item_price`: Maximum price of an item in the order.
- `total_onshift_dashers`: Total number of dashers available for delivery.
- `total_busy_dashers`: Total number of dashers currently busy with other orders.
- `total_outstanding_orders`: Total number of outstanding orders awaiting delivery.
- `estimated_order_place_duration`: Estimated time taken to place the order.
- `estimated_store_to_consumer_driving_duration`: Estimated driving duration from store to consumer.

## Data Exploration and Preprocessing

The initial steps include data loading, cleaning, and exploration:

- Loading the dataset and checking for missing values.
- Removing rows with missing values.
- Calculating delivery duration in seconds.
- Visualizing the distribution of various features such as market ID, order protocols, store categories, delivery days, and months.
- Identifying and handling outliers in the data.

## Feature Engineering

Feature engineering involves creating new features or transforming existing ones to improve model performance:

- One-hot encoding categorical variables such as market ID, order protocol, and store primary category.
- Creating harmonic calendar features to capture time-related patterns.
- Generating time-based harmonic features to capture cyclic patterns in time.
- Combining all features into a single DataFrame for modeling.

## Model Building and Evaluation

The final steps include building predictive models and evaluating their performance:

- Splitting the data into training and testing sets.
- Training an XGBoost regressor model to predict delivery duration.
- Tuning hyperparameters using cross-validation and grid search.
- Evaluating the model performance using metrics like RMSE (Root Mean Squared Error) and MAE (Mean Absolute Error).
- Visualizing the model predictions compared to actual delivery durations.

## Conclusion

The predictive model developed in this project can effectively estimate the delivery duration of food orders, which can be valuable for optimizing delivery operations and enhancing customer experience. Further improvements and optimizations can be explored to enhance the model's performance and scalability.
