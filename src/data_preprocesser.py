# -*- coding: utf-8 -*-

import pandas as pd

import numpy as np

from scipy.stats import skew
from scipy.special import boxcox1p


class preprocessing_data:
    def __init__(self):
        self.dict_filling_value = {}
        
    def convert_data_type(self, df, var, to_num = True):
        if to_num:
            df[var] = df[var].astype(float)
            print(f"Done: Convert {var} to float data type")
        else:
            df[var] = df[var].astype(str)
            print(f"Done: Convert {var} to string data type")
        
    def data_imputation(self, df, cols, fill_values):
        df[cols] = df[cols].fillna(fill_values)
        print(f"Done: Incert missing values for {cols} with {fill_values}")
        
    def filling_missing_values(self, df):
        fill_none = ['FireplaceQu', 'MasVnrType', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']
        self.data_imputation(df, fill_none, 'None')

        # we will fill zeros to the following features if missed
        fill_zero = ['TotalBsmtSF', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
        self.data_imputation(df, fill_zero, 0)
        
        ## for the following features, we will fill the missing values with mostly used value
        fill_mostly_used = ['MSZoning', 'Utilities', 'Functional', 'Electrical', 'KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']
        if len(self.dict_filling_value) == 0:
            for var in fill_mostly_used:
                mostly_used = df[var].mode()[0]
                self.data_imputation(df, [var], mostly_used)
                self.dict_filling_value[var] = mostly_used
        else:
            for k, v in self.dict_filling_value.items():
                self.data_imputation(df, [k], v)
    
    def drop_cols_with_high_missing_perc(self, df, threshold = 0.8):
        '''
        with high missing percentage in the training set, we should drop those cols
        '''
        df_null_counts = df.isnull().sum().reset_index().rename(columns = {0:'null_counts'})
        df_null_counts['null_perc'] = df_null_counts['null_counts'] / df.shape[0]
        # we will drop high missing percentage features (over 80% missing values)
        self.high_missing_cols = df_null_counts[df_null_counts['null_perc']>threshold]['index'].tolist()
        print(f"Done: Found columns {self.high_missing_cols} with high missing percentages")


    def get_high_corr_features(self, df, threshold = 0.5):
        '''
        We will use training set only to find the high correlation features to use
        '''
        corrmat = df.corr()
        corr_selector = corrmat['SalePrice'].abs().sort_values(ascending=False)
        # we want to keep high corr features, but not target itself
        self.high_corr_cols = corr_selector[(corr_selector.values>=threshold) & (corr_selector.values!=1)].index
        print(f"Done: Found {len(self.high_corr_cols)} features with high correlation of 50% or higher with target")
    
    def fix_skewness_for_features(self, df, threshold = 0.75):
        '''
        With some visulization, we can notice that there are many numerical features with high skewness,
        here, we use skew function in scipy.status to help determine skewed_features
        we can fix the skewness by applying box cox transfermation 
        '''
        numeric_feats = df.dtypes[df.dtypes != "object"].index
        # Check the skew of all numerical features
        skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
        skewness = pd.DataFrame({'Skew' :skewed_feats})
        skewness = skewness[abs(skewness) > threshold]
        print("Found {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

        self.skewed_features = skewness.index
        lam = 0.15
        for feat in self.skewed_features:
            #all_data[feat] += 1
            df[feat] = boxcox1p(df[feat], lam)
            print(f"Done: Used Box Cox transform to fixed the skewness for {feat}")
        
        
    def fix_skewness_for_target(self, train_y):
        '''
        for the target var, we will perform log transformation to fix the skewness
        '''
        print("Done: Used log transform to fixed the skewness for the target")
        return np.log(train_y)

    def convert_cate_var_to_dummy(self, df):
        '''
        In order to fit in most of models, we need to convert categorical vars to dummy (numerical features)
        '''
        df = pd.get_dummies(df)
        print("Done: Convert category vars to numerical")
    
    def get_data_ready(self, df_data, training_set = True):
        if training_set == True:
            train_y = df_data.SalePrice
            df_data.drop(['Id'], axis = 1, inplace = True)
            # we will convert the total basement sf to numerical values
            self.convert_data_type(df_data, 'TotalBsmtSF')
            self.filling_missing_values(df_data)
            # we will change the testset to True when we pass the test set
            self.drop_cols_with_high_missing_perc(df_data)

            df_data = df_data.drop(self.high_missing_cols, axis = 1)
            print(f"Done: Drop columns {self.high_missing_cols} due to high missing percentages")

            self.get_high_corr_features(df_data)
            ## we concat training and test sets, prior to that, we will drop target var
            df_data = df_data[self.high_corr_cols]
            print("Done: Only use the high correlation cols for model training")

            self.fix_skewness_for_features(df_data)
            # if there are selected featreus are categorical, we will convert them to dummies
            if df_data.select_dtypes(include = ['object']).shape[1]>0:
                self.convert_cate_var_to_dummy(df_data)
            else:
                print('We have slected all numerical features for model training')
                
            train_y = self.fix_skewness_for_target(train_y)
            return df_data, train_y
        else:
            df_data.drop(['Id'], axis = 1, inplace = True)
            self.convert_data_type(df_data, 'TotalBsmtSF')
            self.filling_missing_values(df_data)
            
            df_data = df_data.drop(self.high_missing_cols, axis = 1)
            print(f"Done: Drop columns {self.high_missing_cols} due to high missing percentages")
            df_data = df_data[self.high_corr_cols]
            print("Done: Only use the high correlation cols for model training")
       
            self.fix_skewness_for_features(df_data)
            # if there are selected featreus are categorical, we will convert them to dummies
            if df_data.select_dtypes(include = ['object']).shape[1]>0:
                self.convert_cate_var_to_dummy(df_data)
            else:
                print('We have slected all numerical features for model training')
            return df_data
        
if __name__ == '__main__':
    print('This is data preprocessor class, not main function')