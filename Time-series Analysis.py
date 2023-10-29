# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 20:32:17 2023

@author: Yigitalp
"""
# Import libraries
from datetime import timedelta
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from xgboost import plot_importance
from xgboost import XGBRegressor
from xgboost import plot_tree
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
    data.plot(legend=False, figsize=(12, 5))
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

# Re-plot transformed dataset
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

# Let's check electricity consumption per hour and month


def consumption_plot(feature, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=df_final, x=feature, y='PJMW_MW', ax=ax)
    ax.set_title(title)


consumption_plot('hour', 'MW per Hour')
consumption_plot('month', 'MW per Month')

# MinMax Normalization
features = [col for col in df_final.columns if col != 'PJMW_MW']
for col in features:
    df_final[col] = (df_final[col] - df_final[col].min()) / \
        (df_final[col].max()-df_final[col].min())

# Calculate correlation
corr = abs(df_final.corrwith(df_final['PJMW_MW']))
corr = corr.drop(index='PJMW_MW')
corr.name = 'Corr_scores'

# Create X & y
X = df_final.drop(columns=['PJMW_MW'])
y = df_final['PJMW_MW']

# Calculate MI score
MI_score = pd.Series(mutual_info_regression(X, y), name='MI_scores')
MI_score.index = X.columns

# Feature selection per weighted score
feature_scores = pd.concat([corr, MI_score], axis=1, names=[
                           'Corr_scores', 'MI_scores'])
for col in feature_scores:
    feature_scores[col] = (feature_scores[col]-feature_scores[col].min()) / \
        (feature_scores[col].max()-feature_scores[col].min())
alpha = 0.5
feature_scores['Weighted_scores'] = alpha * \
    feature_scores['Corr_scores']+(1-alpha)*(feature_scores['MI_scores'])
selected_features = [
    col for col in feature_scores.index if feature_scores.loc[col, 'Weighted_scores'] >= alpha]
X = X[selected_features]

# perform train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, train_size=0.8, shuffle=False)

# Let's plot training and test datasets' targets after 80/20 split w/o shuffle
fig, ax = plt.subplots(figsize=(12, 5))
y_train.plot(ax=ax, label='Training Set')
y_test.plot(ax=ax, label='Test Set')
vertical_line = y_test.index.min()
ax.axvline(vertical_line, color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])

# Train a model
model = XGBRegressor(n_estimators=1000)
model.fit(X_train, y_train,
          eval_set=[(X_train, y_train), (X_test, y_test)],
          early_stopping_rounds=50,
          verbose=False)

# Let's see which features are important
plot_importance(model, height=0.9)

# Let's plot decision tree
fig, ax = plt.subplots(figsize=(170, 170))
plot_tree(model, ax=ax, rankdir='LR')
plt.savefig("decision_tree.pdf")

# Time to make predictions
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)
r_square = r2_score(y_test, predictions)

y_test = y_test.reset_index()
y_test['Predictions'] = predictions
y_test = y_test.set_index('Datetime')
df_final_join = pd.merge(
    df_final, y_test['Predictions'], on='Datetime', how='left')

# Plot predictions on actuals
fig, ax = plt.subplots(figsize=(12, 5))
df_final_join['PJMW_MW'].plot(ax=ax, label='Actuals')
df_final_join['Predictions'].plot(ax=ax, label='Predictions')
vertical_line = y_test.index.min()
ax.axvline(vertical_line, color='black', ls='--')
ax.legend(['Actuals', 'Predictions'])

# Zoom in May 2016-2017-2018


def zoom_plot(lower, upper, title):
    fig, ax = plt.subplots(figsize=(12, 5))
    df_final_join['PJMW_MW'].plot(
        ax=ax, style='-', label='Actuals')
    df_final_join['Predictions'].plot(
        ax=ax, style='.', label='Predictions')
    ax.set_xbound(lower=lower, upper=upper)
    ax.set_title(title)
    ax.legend(['Actuals', 'Predictions'])


zoom_plot('05-01-2016', '05-31-2016',
          'May 2016 Actual vs Predictions')
zoom_plot('05-01-2017', '05-31-2017',
          'May 2017 Actual vs Predictions')
zoom_plot('05-01-2018', '05-31-2018',
          'May 2018 Actual vs Predictions')

# Best and Worst Performers
# First, calculate Absolute Percentage Error for test set's target
y_test['APE'] = 100 * \
    (abs(y_test['PJMW_MW']-y_test['Predictions'])/y_test['PJMW_MW'])

# Find best and worst performer
best_performer = y_test.idxmin()['APE']
best_performer_add_one_day = best_performer + timedelta(days=1)
worst_performer = y_test.idxmax()['APE']
worst_performer_add_one_day = worst_performer + timedelta(days=1)

# Plot best and worst performers
zoom_plot(best_performer, best_performer_add_one_day,
          'Best Actual vs Predictions')
zoom_plot(worst_performer, worst_performer_add_one_day,
          'Worst Actual vs Predictions')
#%%
# More analysis per different time-stamps
y_test_final = create_features(y_test)
analysis_year = y_test_final.groupby(['year'])['APE'].agg(
    ['min', 'mean', 'max', 'std', 'sum'])
analysis_year_month = y_test_final.groupby(['year', 'month'])['APE'].agg([
    'min', 'mean', 'max', 'std', 'sum'])
analysis_year_month['mean'].plot(figsize=(12, 5))

# Further analysis can be performed by applying univariate, bivariate, and multivariate analysis
# on features for better feature selection.
# Holiday periods can be flagged so that it can be an another feature.
