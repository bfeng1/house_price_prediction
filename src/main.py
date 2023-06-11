# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import data_preprocesser
#%%
if __name__ == '__main__':
    df_data = pd.read_csv('data/train.csv')
    data_processer = preprocessing_data()
    X, target = data_processer.get_data_ready(df_data, True)
    
    model = RandomForestRegressor(max_depth=14, max_features = 3)
    model.fit(X, target)

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
        for feature in X.columns.tolist():
        
            # gather user inputs
            user_input = input(f"input {name_map[feature]}: ")
            while user_input.isnumeric() == False:
                user_input = input(f"please use number only {name_map[feature]}: ")
            test_X[feature] = float(user_input)
        test_X = pd.DataFrame(test_X, index=[0])
        predicted = model.predict(test_X)
        predicted_dollars_value = round(np.exp(predicted)[0], 2)
        print(f'Predicted house price based on given features: $ {predicted_dollars_value}')
        start = input('try it out [Y/N]: ').upper()
        while (start != 'Y') & (start != 'N'):
            start = input('try it out [Y/N]: ').upper()
            
        print('Thank you for using!')