DoorDash Delivery Time Prediction This repository contains a notebook for predicting the total time in seconds between the submission of an order (created_at) and its delivery (actual_delivery_time) using historical data from DoorDash. <br />

Data Description The dataset historical_data.csv contains a subset of deliveries received at DoorDash in early 2015 across various cities. Each row represents a unique delivery, with columns corresponding to different features. Notable columns include: <br />

market_id: The city/region where DoorDash operates. <br />
created_at: Timestamp in UTC when the order was submitted. <br />
actual_delivery_time: Timestamp in UTC when the order was delivered. <br />
Store Features store_id: Unique identifier for the restaurant. <br />
store_primary_category: Cuisine category of the restaurant. <br />
order_protocol: Protocol ID for order submission. <br />
total_items: Total number of items in the order. <br />
subtotal: Total order value in cents. <br />
num_distinct_items: Number of distinct items in the order. <br />
min_item_price: Price of the cheapest item. <br />
max_item_price: Price of the most expensive item. <br />
total_onshift_dashers: Number of available dashers within 10 miles. <br />
total_busy_dashers: Number of dashers currently working on an order. <br />
total_outstanding_orders: Number of orders being processed within 10 miles. <br />

Model Used XGBoost was employed to predict the total delivery time. <br />
Dashboard A dashboard created using Tableau is available here for visualizing the data:- https://public.tableau.com/app/profile/pronata.datta/viz/DoordashData/Dashboard1?publish=yes
<br />
Note Both historical_data.csv and data_to_predict.json include noise to obfuscate certain business details. Evaluation of the model's performance will be done on this noisy, artificial dataset.
<br />
For further details, refer to the notebook provided in this repository.
