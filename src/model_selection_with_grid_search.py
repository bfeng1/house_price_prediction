import pandas as pd

import numpy as np

from scipy.stats import skew
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.special import boxcox1p
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

import data_preprocesser

def grid_search(model, name, forest_params, train_X, train_y):
    model_cv = GridSearchCV(model, forest_params, cv = 5, scoring = 'neg_mean_squared_error')
    model_cv.fit(train_X, train_y)
    best_model_score = model_cv.best_score_
    best_model_params = model_cv.best_params_
    print(f'for {name}, we have the best params to use as {best_model_params} with a score of {best_model_score}')
    return best_model_params, best_model_score

if __name__ == '__main__':
    df_data = pd.read_csv('data/train.csv')
    data_processer = preprocessing_data()
    df_data, target = data_processer.get_data_ready(df_data, True)
    
    train_X, test_X, train_y, test_y = train_test_split(df_data, target, test_size=0.2)
    
    model_comperison = []
    # perform grid search to find the best performing models
    name = 'Ridge Regression'
    forest_params = [{'alpha': [100, 10, 1, 0.1, 0.01]}]
    params, score = grid_search(Ridge(), name, forest_params, train_X, train_y)
    model_comperison.append((name, params, score))
    
    name = 'Lasso Regression'
    forest_params = [{'alpha': [100, 10, 1, 0.1, 0.01]}]
    params, score = grid_search(Lasso(), name, forest_params, train_X, train_y)
    model_comperison.append((name, params, score))
    
    name = 'Support Vector Regression'
    forest_params = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C':[100, 10, 1, 0.1, 0.01]}]
    params, score = grid_search(SVR(), name, forest_params, train_X, train_y)
    model_comperison.append((name, params, score))
    
    # perform grid search to find the best performing models
    name = 'Random Forest Regression'
    forest_params = [{'max_depth': list(range(5, 15)), 'max_features': list(range(1,14))}]
    params, score = grid_search(RandomForestRegressor(), name, forest_params, train_X, train_y)
    model_comperison.append((name, params, score))
    
    # pitch the best model to use based on the grid search
    model_comperison.sort(key = lambda x: x[2])
    best_model_to_use = model_comperison[-1]
    print(best_model_to_use)
    
    # build the model using the parameters above with the selected model
    model = RandomForestRegressor(max_depth=14, max_features = 3)
    # fit and predict the results for the test data
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    print(f"with the chosen best model, we got a MSE of {mean_squared_error(test_y, y_pred)}")