# House Price Prediction

## A bit about me
🚀 Hi there! I'm Bin Feng, a Business Intelligence Engineer with a burning passion for all things Data Science and Machine Learning. I thrive on the thrill of exploring data, extracting insights, and turning them into actionable strategies.

📊 My journey in this field has been incredible, but I'm always hungry for more knowledge and skills. I firmly believe that continuous learning is the key to staying at the forefront of this dynamic industry. That's why I'm constantly seeking opportunities to sharpen my skills and delve into advanced models.

🤝 Collaboration is at the heart of my work ethic. I'm eager to team up with like-minded individuals to create something truly exceptional. Whether it's a groundbreaking project or a fascinating experiment, I'm all ears for fresh ideas and open to any advice or suggestions that can elevate our work.

💡 Let's innovate, explore, and make a positive impact together. Feel free to reach out, and let's embark on this exciting journey of data-driven discovery!

Thanks for connecting! 🌟

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
```Mean Squared Error: 0.026993211114126944```

## Addtional Info
* [Kaggle Project Link](https://www.kaggle.com/code/binfeng2021/house-price-prediction)
* [Author LinkedIn Bin Feng](https://www.linkedin.com/in/bfeng1/)

