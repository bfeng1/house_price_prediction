# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

from scipy.stats import skew
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from scipy.special import boxcox1p
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

## data cleaning
class house_price_prediction:
    def __init__(self, df_train):
        df_train.drop("Id", axis = 1, inplace = True)
        self.df_train = df_train
        self.train_y = df_train.SalePrice
        
    def convert_data_type(self, var, to_num = True):
        if to_num:
            self.df_train[var] = self.df_train[var].astype(float)
            print(f"Done: Convert {var} to float data type")
        else:
            self.df_train[var] = self.df_train[var].astype(str)
            print(f"Done: Convert {var} to string data type")
        
    def data_imputation(self, cols, fill_values):
        self.df_train[cols] = self.df_train[cols].fillna(fill_values)
        print(f"Done: Incert missing values for {cols} with {fill_values}")
        
    def filling_missing_values(self):
        fill_none = ['FireplaceQu', 'MasVnrType', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']
        self.data_imputation(fill_none, 'None')

        # we will fill zeros to the following features if missed
        fill_zero = ['TotalBsmtSF', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        self.data_imputation(fill_zero, 0)
        
        ## for the following features, we will fill the missing values with mostly used value
        fill_mostly_used = ['MSZoning', 'Utilities', 'Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
        for var in fill_mostly_used:
            mostly_used = self.df_train[var].mode()[0]
            self.data_imputation([var], mostly_used)
    
    def drop_cols_with_high_missing_perc(self, threshold = 0.8):
        '''
        with high missing percentage in the training set, we should drop those cols
        '''
        df_null_counts = self.df_train.isnull().sum().reset_index().rename(columns = {0:'null_counts'})
        df_null_counts['null_perc'] = df_null_counts['null_counts'] / self.df_train.shape[0]
        # we will drop high missing percentage features (over 80% missing values)
        self.high_missing_cols = df_null_counts[df_null_counts['null_perc']>threshold]['index'].tolist()
        print(f"Done: Found columns {self.high_missing_cols} with high missing percentages")

        self.df_train = self.df_train.drop(self.high_missing_cols, axis = 1)
        print(f"Done: Drop columns {self.high_missing_cols} due to high missing percentages")

    def get_high_corr_features(self, threshold = 0.5):
        '''
        We will use training set only to find the high correlation features to use
        '''
        corrmat = self.df_train.corr()
        corr_selector = corrmat['SalePrice'].abs().sort_values(ascending=False)
        # we want to keep high corr features, but not target itself
        self.high_corr_cols = corr_selector[(corr_selector.values>=threshold) & (corr_selector.values!=1)].index
        print(f"Done: Found {len(self.high_corr_cols)} features with high correlation of 50% or higher with target")
    
    def fix_skewness_for_features(self, threshold = 0.75):
        '''
        With some visulization, we can notice that there are many numerical features with high skewness,
        here, we use skew function in scipy.status to help determine skewed_features
        we can fix the skewness by applying box cox transfermation 
        '''
        numeric_feats = self.df_train.dtypes[self.df_train.dtypes != "object"].index
        # Check the skew of all numerical features
        skewed_feats = self.df_train[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness = skewness[abs(skewness) > threshold]
        print("Found {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        self.skewed_features = skewness.index
        lam = 0.15
        for feat in self.skewed_features:
            #all_data[feat] += 1
            self.df_train[feat] = boxcox1p(self.df_train[feat], lam)
            print(f"Done: Used Box Cox transform to fixed the skewness for {feat}")
        
        
    def fix_skewness_for_target(self):
        '''
        for the target var, we will perform log transformation to fix the skewness
        '''
        self.train_y = np.log(self.train_y)
        print("Done: Used log transform to fixed the skewness for the target")

    def convert_cate_var_to_dummy(self):
        '''
        In order to fit in most of models, we need to convert categorical vars to dummy (numerical features)
        '''
        self.df_train = pd.get_dummies(self.df_train)
        print("Done: Convert category vars to numerical")
        
    def get_training_data_ready(self):
        # we will convert the total basement sf to numerical values
        self.convert_data_type('TotalBsmtSF')
        self.filling_missing_values()
        # we will change the testset to True when we pass the test set
        self.drop_cols_with_high_missing_perc()
        self.get_high_corr_features()
        
        ## we concat training and test sets, prior to that, we will drop target var
        self.df_train = self.df_train[self.high_corr_cols]
        print("Done: Only use the high correlation cols for model training")
        
        self.fix_skewness_for_features()
        self.fix_skewness_for_target()
        
        # if there are selected featreus are categorical, we will convert them to dummies
        if self.df_train.select_dtypes(include = ['object']).shape[1]>0:
            self.df_train = self.convert_cate_var_to_dummy()
        else:
            print('We have slected all numerical features for model training')
            
        # after we preprocess the data, we will get train_X, train_y
        print("Done: We have created df_train, train_y, they are ready for model training!")
        
    def rmsle_cv(self, model, name, n_folds = 5):
        '''
        We use this function to define cross validation method
        N fold + RMSLE score
        '''
        kf = KFold(n_folds, shuffle = True, random_state = 42).get_n_splits(self.df_train)
        rmse = np.sqrt(-cross_val_score(model, self.df_train.values, self.train_y.values, scoring = "neg_mean_squared_error", cv = kf))
        print('score for the model {} is {} ({})'.format(name, round(rmse.mean(),4), round(rmse.std(),4)))
        return rmse.mean()
    
    def create_pipelinefor_model_comparison(self):
        models_scores = []
        
        LR = make_pipeline(StandardScaler(), LinearRegression())
        score_lr = self.rmsle_cv(LR, 'Linear Regression')
        models_scores.append((LR, score_lr, 'Linear Regression'))
        
        RR = make_pipeline(StandardScaler(), Ridge(alpha = 0.002))
        score_rr = self.rmsle_cv(RR, 'Ridge Regression')
        models_scores.append((RR, score_rr, 'Ridge Regression'))
        
        lasso = make_pipeline(StandardScaler(), Lasso(alpha =0.0005, random_state=1))
        # lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1))
        score_lasso = self.rmsle_cv(lasso, 'Lasso')
        models_scores.append((lasso, score_lasso, 'Lasso Regression'))
        
        svr = make_pipeline(StandardScaler(), SVR(C = 1.0, epsilon = 0.2))
        score_svr = self.rmsle_cv(svr, 'SVR')
        models_scores.append((svr, score_svr, 'Support Vector Regression'))
        
        RFR = make_pipeline(StandardScaler(), RandomForestRegressor(max_depth = 35, random_state = 0))
        score_rfr = self.rmsle_cv(RFR, 'Random Forest Regressor')
        models_scores.append((RFR, score_rfr, 'Random Forest Regressor'))
        
        model_xgb = xgb.XGBRegressor(colsample_bytree = 0.4603, 
                                     gamma = 0.0468, 
                                     learning_rate = 0.05, 
                                     max_depth = 3, 
                                    min_child_weight = 1.7817,
                                    reg_alpha = 0.4640,
                                    reg_lambda = 0.8571,
                                    subsample = 0.5213,
                                    silent=1,
                                    randome_state =7,
                                    nthread = -1)
        score_xgb = self.rmsle_cv(model_xgb, 'XGBoost')
        models_scores.append((model_xgb, score_xgb, 'XGBoost'))
        
        self.models_scores = sorted(models_scores, key = lambda x: x[1])
        print('Done: Model Selection')
        print('The best model we got is {}'.format(self.models_scores[0]))
    
    def get_test_data_ready(self):
        '''
        The function take one row of test data as input
        perform the same cleaning and preprocessing as training set
        '''
        # first, convert TotalBsmtSF to numerical values
        self.test_X['TotalBsmtSF'] = self.test_X['TotalBsmtSF'].astype(float)
        # since we have all numerical features, fill 0 for any missing values
        self.test_X = self.test_X.fillna(0)
        # apply same type of skewness fix as training set
        for feat in self.skewed_features:
            self.test_X[feat] = ((1+self.test_X[feat].values[0]) ** 0.15 - 1)/0.15
        
    def predict(self, test_dict):
        self.test_X = pd.DataFrame(test_dict, index=[0])
        self.get_test_data_ready()
        (chosen_model, score, name) = self.models_scores[0]
        chosen_model.fit(self.df_train.values, self.train_y.values)
        predicted = chosen_model.predict(self.test_X)
        predicted_dollars_value = round(np.exp(predicted)[0], 2)
        print(f'Predicted house price based on given features: $ {predicted_dollars_value}')
        
if __name__ == '__main__':
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    # initial the data preparation object
    house_price_predictor = house_price_prediction(df_train)
    # get training data ready, cleaning and preprocessing
    house_price_predictor.get_training_data_ready()
    # select model and training it, find the best performed one
    house_price_predictor.create_pipelinefor_model_comparison()
    print('The model is ready to use, now lets try it out!')
    
    ###
    name_map = {}
    name_map['OverallQual'] = 'Rates the overall material and finish of the house'
    name_map['YearBuilt'] = 'Original construction date'
    name_map['YearRemodAdd'] = 'Remodel date (same as construction date if no remodeling or additions)'
    name_map['TotalBsmtSF'] = 'Total square feet of basement area'
    name_map['GrLivArea'] = 'Above grade (ground) living area square feet'
    name_map['FullBath'] = 'Full bathrooms above grade'
    name_map['TotRmsAbvGrd'] = 'Total rooms above grade (does not include bathrooms)'
    name_map['GarageCars'] = 'Size of garage in car capacity'
    name_map['GarageArea'] = 'Size of garage in square feet'
    name_map['1stFlrSF'] = 'First Floor square feet'
                
    start = input('try it out [Y/N]: ').upper()
    while (start != 'Y') & (start != 'N'):
        start = input('try it out [Y/N]: ').upper()
    
    while start == 'Y':
        test_X = {}
        for feature in house_price_predictor.high_corr_cols:
        
            # gather user inputs
            user_input = input(f"input {name_map[feature]}: ")
            while user_input.isnumeric() == False:
                user_input = input(f"please use number only {name_map[feature]}: ")
            test_X[feature] = float(user_input)
        house_price_predictor.predict(test_X)
        
        start = input('try it out [Y/N]: ').upper()
        while (start != 'Y') & (start != 'N'):
            start = input('try it out [Y/N]: ').upper()
            
        print('Thank you for using!')