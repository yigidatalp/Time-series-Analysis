# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:32:17 2023

@author: Yigitalp
"""
# Import libraries
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from xgboost import plot_importance
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

# Load dataset and sort it by date
df = pd.read_csv('PJMW_hourly.csv', index_col=['Datetime'], parse_dates=True)
df = df.sort_index()

# Check dataset if there are missing values
df.info()


def check_plot(data, x):
    data.plot(color='red', legend=False, figsize=(12, 5))
    fig, ax = plt.subplots(2, 1, figsize=(12, 5))
    sns.histplot(data=data, x=x, ax=ax[0]).set(xlabel=None)
    sns.boxplot(data=data, x=x, ax=ax[1])


# Plot data to check if the shape of dataset isn't normal or there are outliers
check_plot(df, 'PJMW_MW')

# According to boxplot there are outliers in the dataset: let's remove them
iqr = df['PJMW_MW'].quantile(0.75) - df['PJMW_MW'].quantile(0.25)
min_whisker_value = df['PJMW_MW'].quantile(0.25)-1.5*iqr
max_whisker_value = df['PJMW_MW'].quantile(0.75)+1.5*iqr
df_transformed = df[(df['PJMW_MW'] >= min_whisker_value) &
                    (df['PJMW_MW'] <= max_whisker_value)]
df_transformed = df_transformed.sort_index()

# Re-plot transformed dataset and sort it by date
check_plot(df_transformed, 'PJMW_MW')

# Feature creation


def create_features(df):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dayofmonth'] = df['date'].dt.day
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['weekofyear'] = df['date'].dt.weekofyear
    df = df.drop('date', axis=1)
    return df


df_final = create_features(df_transformed)

# Create X & y and perform train_test_split
X = df_final.iloc[:, 1:]
y = df_final['PJMW_MW']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, shuffle=False)


# Train a model
model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
                     max_depth=3, min_child_weight=0,
                     gamma=0, subsample=0.7,
                     colsample_bytree=0.7,
                     objective='reg:squarederror', nthread=-1,
                     scale_pos_weight=1, seed=27,
                     reg_alpha=0.00006)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=50,
          verbose=False)

# Let's see which features are important
plot_importance(model, height=0.9)

# Time to predict
predictions = model.predict(X_test)
mae = mean_absolute_error(predictions, y_test)
mape = mean_absolute_percentage_error(predictions, y_test)

y_test = y_test.reset_index()
y_test['Predictions'] = predictions
y_test = y_test.set_index('Datetime')
df_final_join = pd.merge(df_final, y_test, on='Datetime', how='left')

# Plot predictions on actuals
df_final_join[['PJMW_MW_x', 'Predictions']].plot(
    color=['red', 'green'], legend=False, figsize=(12, 5))

# Zoom in May 2016-2017-2018


def zoom_plot(df, lower, upper, title):
    fig, ax = plt.subplots(1, figsize=(12, 5))
    df[['PJMW_MW_x', 'Predictions']].plot(
        ax=ax, style=['-', '.'], color=['red', 'green'], figsize=(12, 5))
    ax.set_xbound(lower=lower, upper=upper)
    ax.set_title(title)


zoom_plot(df_final_join, '05-01-2016', '05-31-2016',
          'May 2016 Actual vs Predictions')
zoom_plot(df_final_join, '05-01-2017', '05-31-2017',
          'May 2017 Actual vs Predictions')
zoom_plot(df_final_join, '05-01-2018', '05-31-2018',
          'May 2018 Actual vs Predictions')

# Best and Worst Performers
# First, calculate Absolute Percentage Error for test set's target
y_test['APE'] = 100 * \
    (abs(y_test['PJMW_MW']-y_test['Predictions'])/y_test['PJMW_MW'])

# Plot best and worst performers
zoom_plot(df_final_join, '06-28-2016', '06-29-2016',
          'Best Actual vs Predictions')
zoom_plot(df_final_join, '02-25-2017', '02-26-2017',
          'Worst Actual vs Predictions')
#%%
# More analysis per different time-stamps
y_test_final = create_features(y_test)
mae_year = y_test_final.groupby(['year'])['APE'].mean()
mae_year_month = y_test_final.groupby(['year', 'month'])['APE'].mean()
mae_year_quarter_month = y_test_final.groupby(
    ['year', 'quarter', 'month'])['APE'].mean()
mae_year_quarter_month.plot(figsize=(12, 5))
