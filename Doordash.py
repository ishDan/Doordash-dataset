import pandas as pd
import plotly.express as px

data = pd.read_csv('historical_data.csv', parse_dates=['created_at','actual_delivery_time'])
# to_predict = pd.read_json('data_to_predict.json',lines=True)
# info = pd.read_table('data_description.txt')

# Duplicate loading of data for later use

raw = pd.read_csv('historical_data.csv', parse_dates=['created_at','actual_delivery_time'])

# Define a function to display the first 5 rows of a DataFrame

def display(i):
  return i.head(5)

display(data)

# display(to_predict)

# info

# Calculate delivery duration in seconds

data['duration'] = (data['actual_delivery_time']-data['created_at']).dt.total_seconds()

# Check for missing values

data.isna().sum()

# Remove rows with missing values

data.dropna(inplace=True)

# Display summary statistics

data.describe()

# Visualize the distribution of cities/regions (market_id)

market_id_distribution = px.bar(data.market_id.value_counts(), color=data.market_id.unique(), labels={'value': 'Count', 'index': 'Market ID'})
market_id_distribution.update_layout(title='Distribution of Cities/Regions (Market ID)')
market_id_distribution.show()

# Visualize the distribution of order protocols

order_protocol_distribution = px.bar(data.order_protocol.value_counts().sort_values(ascending=False), color=data.order_protocol.unique(), labels={'value': 'Count', 'index': 'Order Protocol'})
order_protocol_distribution.update_layout(title='Distribution of Order Protocols')
order_protocol_distribution.show()

# Visualize the distribution of store primary categories

store_category_distribution = px.bar(data.store_primary_category.value_counts(), color=data.store_primary_category.unique(), labels={'value': 'Count', 'index': 'Store Primary Category'})
store_category_distribution.update_layout(title='Distribution of Store Primary Categories')
store_category_distribution.show()

# Visualize the distribution of deliveries by day of the week. The bar graph will provide clear information about the distribution of deliveries by day of the week.

delivery_day_distribution = px.bar(data.actual_delivery_time.dt.day_name().value_counts(), 
                                   color=data.actual_delivery_time.dt.day_name().unique(),
                                   labels={'value': 'Count', 'index': 'Day of the Week'})
delivery_day_distribution.update_layout(title='Distribution of Deliveries by Day of the Week')
delivery_day_distribution.show()

# Visualize the distribution of deliveries by month

delivery_month_distribution = px.bar(data.actual_delivery_time.dt.month_name().value_counts(), 
                                     color=data.actual_delivery_time.dt.month_name().unique(),
                                     labels={'value': 'Count', 'index': 'Month'})
delivery_month_distribution.update_layout(title='Distribution of Deliveries by Month')
delivery_month_distribution.show()


# Visualize the distribution of deliveries by day of the month
delivery_day_of_month_distribution = px.bar(data.actual_delivery_time.dt.day.value_counts(), 
                                            color=data.actual_delivery_time.dt.day.unique(),
                                            labels={'value': 'Count', 'index': 'Day of the Month'})
delivery_day_of_month_distribution.update_layout(title='Distribution of Deliveries by Day of the Month')
delivery_day_of_month_distribution.show()


# Identify holidays within the data range

from pandas.tseries.holiday import USFederalHolidayCalendar as calendar

start = raw.created_at.dt.date.min()
end = raw.created_at.dt.date.max()
holidays = calendar().holidays(start=start, end=end, return_name=True)
print(f'Holidays in range ({start} - {end}):\n', holidays)


# Extend the date range for holiday detection

dates_with_margin = pd.date_range(
        start - pd.DateOffset(months=1),
        end + pd.DateOffset(months=4))

dates_with_margin

# Get holidays within the extended date range

holidays = calendar().holidays(
    start=dates_with_margin.min(),
    end=dates_with_margin.max(), return_name=True)
holidays

# Count the number of holidays

is_holiday = pd.Series(pd.Series(dates_with_margin).isin(holidays).values, index=dates_with_margin)
is_holiday.sum()

# so we count them as outliers

data.duration.quantile(0.9999)/3600

# cleaning data

raw.columns

# Identify outliers and clean the data
raw['duration'] = (raw.actual_delivery_time - raw.created_at).dt.total_seconds()

# date_outlier = pd.to_datetime('2014-12-31')
duration_outlier = 60*60*6  # 6 hours

# Remove outliers and rows with missing values in key columns
# cleaned_raw = raw[
#     raw.created_at > date_outlier
# ][
cleaned_raw = raw[raw.duration < duration_outlier
].dropna(how='any',
         subset=[
             'duration',
             'market_id',
             'store_primary_category',
             'order_protocol',
             'total_onshift_dashers',
             'total_busy_dashers',
             'total_outstanding_orders'
         ])

# Display the size difference before and after cleaning

print("Old dataset size: ", (raw.shape), "----->", "New dataset size: ", (cleaned_raw.shape))
cleaned_raw.head()


# Visualize the relationship between subtotal and delivery duration. 
# By plotting the frequency of delivery durations for each subtotal amount (in dollars), we can observe the typical or usual durations associated with different order subtotals.

cleaned_raw['subtotal_dollars'] = cleaned_raw['subtotal'] / 100 # Converting cents to dollar

cleaned_raw['duration_hours'] = cleaned_raw['duration'] / 3600 # Converting seconds to hours

subtotal_duration_relationship = px.scatter(cleaned_raw, x='subtotal_dollars', y='duration_hours', 
                                            labels={'subtotal_dollars': 'Subtotal ($)', 'duration_hours': 'Delivery Duration (in hours)'})
subtotal_duration_relationship.update_layout(title='Relationship between Subtotal(in dollars) and Delivery Duration')
subtotal_duration_relationship.show()

# Extract basic numerical features

basic_features  = cleaned_raw[[
    'total_items',
    'subtotal',
    'num_distinct_items',
    'min_item_price',
    'max_item_price',
    'total_onshift_dashers',
    'total_busy_dashers',
    'total_outstanding_orders',
    'estimated_order_place_duration',
    'estimated_store_to_consumer_driving_duration',
    ]]

basic_features.head()

# One-hot encode categorical variables

alphabetic_col = cleaned_raw[[
    'market_id',
    'order_protocol',
    'store_primary_category'
]]

#display

alphabetic_col.head()

from sklearn.preprocessing import OneHotEncoder as ohe

encoder = ohe()
encoder.fit(alphabetic_col[['market_id', 'order_protocol', 'store_primary_category']])
encoded_column = encoder.transform(alphabetic_col[['market_id', 'order_protocol', 'store_primary_category']]).toarray()

print(encoded_column.shape)
encoded_values = pd.concat([pd.DataFrame(encoded_column, columns=encoder.get_feature_names_out(['market_id', 'order_protocol', 'store_primary_category']))], axis=1)
encoded_values.head()

# Merge encoded columns with basic features
encoded_table = pd.merge(basic_features, encoded_values, left_index=True, right_index=True, how='left')
encoded_table.head()
encoded_table.shape
encoded_table.head()

# merging encoded_column with cleaned_raw and not resetting the index. Calling it encoded_table

encoded_table = pd.merge(basic_features, encoded_values, left_index=True, right_index=True, how='left')

#display

print("Shape:", encoded_table.shape)
encoded_table.head()


# Create harmonic calendar features

import numpy as np

def harmonic_func(value, period):
  value *= 2 * np.pi / period
  return np.cos(value), np.sin(value)

df = pd.DataFrame(index=data.created_at.dt.normalize().unique())
df['cos_day'], df['sin_day'] = harmonic_func(df.index.day, df.index.days_in_month)
df['cos_week'], df['sin_week'] = harmonic_func(df.index.day_of_week, 7)
df['cos_month'], df['sin_month'] = harmonic_func(df.index.month, 12)
df['cos_quater'], df['sin_quater'] = harmonic_func(df.index.quarter, 4)
df['cos_year'], df['sin_year'] = harmonic_func(df.index.year, 365)
df.head(3)

df['is_holiday'] = df.index.isin(holidays)
dates_extended = pd.date_range(pd.to_datetime(df.index.min()) - pd.DateOffset(months=4), pd.to_datetime(df.index.max()) + pd.DateOffset(months=4))
holiday_ = calendar().holidays(start=dates_extended.min(), end=dates_extended.max())
df['is_holiday'] = df.index.isin(holiday_)
df.head()



df['is_holiday'] = df.index.isin(holidays)
dates_extended = pd.date_range(pd.to_datetime(df.index.min()) - pd.DateOffset(months=4), pd.to_datetime(df.index.max()) + pd.DateOffset(months=4))
holiday_ = calendar().holidays(start=dates_extended.min(), end=dates_extended.max())
df['is_holiday'] = df.index.isin(holiday_)
print(df.shape)
df.head()


# Merge calendar features with the cleaned dataset

calender_features = pd.DataFrame({'normalized_date': cleaned_raw.created_at.dt.normalize()}, index=cleaned_raw.index) \
        .merge(df.fillna(0), left_on='normalized_date', right_index=True) \
        .drop(columns=['normalized_date'])

print(calender_features.shape)
calender_features.head()

# Generate time-based harmonic features

# calculates the total time of the day in minutes, considering both the hour and minute components of the created_at timestamp
min_cos, min_sin = harmonic_func(cleaned_raw.created_at.dt.hour*60 + cleaned_raw.created_at.dt.minute, 24*60*60)

time_features = pd.DataFrame({'min_cos': min_cos, 'min_sin': min_sin}, index=cleaned_raw.index)
time_features.head()


# Combine all features into a single DataFrame

features = pd.concat([basic_features, encoded_table, calender_features, time_features], axis=1)
display(features)


# Removing duplicate columns in features

features = features.loc[:,~features.columns.duplicated()]
display(features)

# Target
target = cleaned_raw.reindex(columns=['duration'])
target.head(3)

# save the prepared dataset

dataset = pd.concat([features,target],axis=1)
dataset.head(3)

# Saving the csv file in my drive

# dataset.to_csv('/content/drive/My Drive/cleaned_data_new.csv')


from sklearn.model_selection import train_test_split, KFold, GridSearchCV
import xgboost
from sklearn.metrics import mean_squared_error, mean_absolute_error

cleaned_df = dataset

cleaned_df.reset_index(drop=True)
print(cleaned_df.index.duplicated().any())

# create a dataframe y that takes duration from clean_df and reindex it
y = cleaned_df['duration'].to_frame()
y = y.reset_index(drop=True)
assert y.isnull().any().any() == False
y

x = cleaned_df.drop(columns=['duration'])
x = x.reset_index(drop=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=2)
for i in [x_train, x_test, y_train, y_test]:
  print(i.shape)
cleaned_df = cleaned_df.reset_index(drop=True)
display(cleaned_df)


basic_features_x = cleaned_df[['total_items',
    'subtotal',
    'num_distinct_items',
    'min_item_price',
    'max_item_price',
    'total_onshift_dashers',
    'total_busy_dashers',
    'total_outstanding_orders',
    'estimated_order_place_duration',
    'estimated_store_to_consumer_driving_duration']]
basic_features_x.head(5)

basic_x_train, basic_x_test, basic_y_train, basic_y_test = train_test_split(basic_features_x, y, test_size=.2, shuffle=True, random_state=2)
basic_x_train = basic_x_train.fillna(0)
basic_x_test = basic_x_test.fillna(0)
for i in [basic_x_train, basic_y_train, basic_x_test, basic_y_test]:
  print(i.shape)

basic_x_train.columns

model = xgboost.XGBRegressor()
history = model.fit(
    basic_x_train, basic_y_train,
    eval_set=[(basic_x_test, basic_y_test)],
    eval_metric=['rmse', 'mae'],
    verbose=0,
    early_stopping_rounds=10)


pd.DataFrame(history.evals_result()['validation_0']).plot()

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# model = xgboost.XGBRegressor(verbose=0, use_label_encoder =False)
# params={
#     'booster': ['gbtree'], #['dart'] - best but shap not supported, #['gbtree', 'gblinear', 'dart'],
#     'objective': ['reg:gamma'], #['reg:squarederror', 'reg:squaredlogerror', 'reg:logistic', 'reg:gamma'],
#     'max_depth': [7], #[3, 4, 5, 6, 7, 10, 15], #--
#     'learning_rate': [0.05, 0.1, 0.15], 
#     'n_estimators': [300, 350, 400],
# #     'min_child_weight': [1, 10], #++
# #     'colsample_bytree': [0.8],
# #     'subsample': [0.75],
# #     'reg_alpha': [0],
# #     'reg_lambda': [2],
# #     'gamma' : [0],
# }
# cv = KFold(5, shuffle=True, random_state=2)
# rs = GridSearchCV(
#     model,
#     params,
#     cv=cv,
#     scoring="neg_mean_squared_error",
#     n_jobs=5,
#     verbose=False)

# rs.fit(
#     x,
#     y,
#     verbose=False)

print('best params:', rs.best_params_)

# best params: {'booster': 'gbtree', 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 350, 'objective': 'reg:gamma'}
best_params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'max_depth': 7,
    'learning_rate': 0.1,
    'n_estimators': 350,
    'verbose':0
}

model = xgboost.XGBRegressor(**best_params)
history = model.fit(
    x_train, y_train,
    eval_set=[(x_test, y_test)],
    eval_metric=['mae', 'rmse'],
    verbose = 0,
    early_stopping_rounds=int(best_params.get('n_estimators', 100) * 0.1))

pd.DataFrame(history.evals_result()['validation_0']).plot()

# Using metrics like RMSE, MAE to see model accuracy

y_true = y_test
y_pred = model.predict(x_test)

print('RMSE: ', np.sqrt(mean_squared_error(y_true, y_pred)))
print('MAE: ', mean_absolute_error(y_true, y_pred))

# Plotting to see model performance

import matplotlib.pyplot as plt

# Plotting the first 50 samples
plot_data = pd.DataFrame({
    'pred': y_pred,
    'real': y_true.duration,
    'diff': (y_pred - y_true.duration).abs(),
}).iloc[:50]

plot_data.plot.bar(y=['pred', 'real', 'diff'], figsize=(15, 5))
plt.xlabel('Sample Index')
plt.ylabel('Duration')
plt.title('Comparison of Predicted and Real Durations')
plt.legend(['Predicted Duration', 'Real Duration'])
plt.show()

