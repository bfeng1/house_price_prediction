# House Price Prediction

## Objectives
The main objective is to develop a house price preditor that uses given training data and advanced regression techniques. 
To find the best performing regression model, I have used the grid search with different parameters to find the best performing model is Random Forest Regression with max_depth as 12 and max_feature as 6 and the MSE on the test dataset is about 0.027.
```
('Random Forest Regression', {'max_depth': 12, 'max_features': 6}, -0.024935079673947064)
with the chosen best model, we got a MSE of 0.026993211114126944
```

## Dataset Description

#### [Ames Housing dataset](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) (Kaggle Competition):
This dataset was compiled by Dean De Cock for use in data science education. It's an incredible alternative for data scientists looking for a modernized and expanded version of the often cited Boston Housing dataset. Based on the model performance comparison, we will use the random forest regression model to perform this task. 

#### Datasets Details
* train.csv - the training set. It has been splited into training data (80%) and testing data (20%) during model selection, and then they are all used to train the final perfoming regression model. 
* data_description.txt - full description of each column, originally prepared by Dean De Cock but lightly edited to match the column names used here

## Project Folder Structure
```
.
├── data
│   ├── data_description.txt
│   └── train.csv
├── requirements.txt
└── src
    ├── __pycache__
    │   └── data_preprocesser.cpython-38.pyc
    ├── data_preprocesser.py
    ├── main.py
    ├── model_selection_with_grid_search.py
    └── train.csv
```

## User Case
### Try the trained model with user inputs
* first, install the required libraries
```pip install -r requirements.txt```
* second, inside the project folder, run the python file in the terminal
```python src/main.py```
* Following the instruction on the terminal, provide user inputs for each feature
* Receive the prediction of the house price

### Reproduce the model selection process with grid search
```python src/model_selection_with_grid_search.py```

## Model Accuracy 
```0.026993211114126944```

## Addtional Info
* [Kaggle Project Link](https://www.kaggle.com/code/binfeng2021/house-price-prediction)
* [Author LinkedIn Bin Feng](https://www.linkedin.com/in/bfeng1/)

