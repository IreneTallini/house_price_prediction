# FILE READING AND MODULE IMPORT

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LassoCV
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv("/home/IT/Documenti/Data_Science/train.csv")
test= pd.read_csv("/home/IT/Documenti/Data_Science/test.csv")
Id = test['Id']

#DATA TIDYING

# Clean outliers
train.drop(train[(train.GrLivArea > 4000 ) & (train.SalePrice < 400000)].index, inplace = True)

# Apply log to SalePrice
train["SalePrice"] = np.log(train["SalePrice"])

# Merge train and test in all_data
ntrain = train.shape[0]
ntest = test.shape[0]
y_train = train.SalePrice.values
all_data = pd.concat((train, test)).reset_index(drop = True)
all_data.drop(['SalePrice'], axis = 1, inplace = True)

# Change some num to object
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['MoSold'] = all_data['MoSold'].apply(str)

# handle NaNs
all_data['Alley'].fillna(value = 'None', inplace = True)

mode = ['MSZoning', 'Utilities', 'Electrical', 'GarageYrBlt', 'SaleType', 'Exterior1st', 'Exterior2nd']
d_mode = {var : all_data[var].mode()[0] for var in mode}

threshold = 0.5 * len(all_data)
all_data.drop(list(filter(lambda var : all_data[var].isnull().sum() > threshold, all_data.keys())), axis = 1, inplace = True)

null = [var for var in all_data.keys() if var not in mode]
d_null = {**{var : 'None' for var in null if var == 'object'}, **{var : 0 for var in null if not var == 'object'}}

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))

d = {**d_mode, **d_null}

all_data.fillna(value = d, inplace = True)

# Remove multicollinearity and 'Id'
corrmat = all_data.corr()
numerical = all_data.select_dtypes(exclude = ['object'])
to_drop = []
for var in numerical:
    for var1 in numerical :
        if (corrmat[var][var1] > 0.7 or corrmat[var][var1] < - 0.7) and not var == var1:
            if np.corrcoef(all_data[var][: ntrain], y_train)[0][1] < np.corrcoef(all_data[var1][: ntrain], y_train)[0][1] :
                to_drop.append(var)
            else :
                to_drop.append(var1)
all_data.drop(to_drop, axis = 1, inplace = True)
all_data.drop('Id', axis = 1, inplace = True)


# FEATURE ENGINEERING
all_data = pd.get_dummies(all_data)

train = all_data[: ntrain]
test = all_data[ntrain :]

# FEATURE SELECTION
count = {var : 0 for var in train.keys()}

# We fixed the random state to avoid oscillations due to the randomization of train_test_split
random_state = [2027,9670,403517,790338,95587,392258,568510,185870,164528,464971,730573,267531,166293,740449,651953,464724,899166,95657,273035,620291]

# 20 rounds of Lasso (with parameters chosen via cross validation). If a feature is chosen by more than 5 rounds, it's selected for fitting.
for i in range(20) :
    X_train, X_test, y_train_new, y_test = train_test_split(train, y_train, random_state = random_state[i])
    model_lasso = LassoCV(alphas = [1, 0.1, 0.001, 0.0005]).fit(X_train, y_train_new)
    not_zero = [train.keys()[i] for i in range(len(train.keys())) if  model_lasso.coef_[i] != 0]
    for var in not_zero :
        count[var] += 1

useful_var = [var for var in count.keys() if count[var] > 5]
all_data_ridotto = pd.DataFrame(data = {i : all_data[i] for i in useful_var})
train_ridotto = all_data_ridotto[0 : ntrain]
test_ridotto = all_data_ridotto[ntrain :]

# MODEL SELECTION 
model_ridge = Ridge(alpha = 22).fit(train_ridotto, y_train)

# SUBMISSION
result_ridge = np.exp(model_ridge.predict(test_ridotto))
rs_ridge = pd.DataFrame(data = {'SalePrice' : result_ridge}, index = Id)
rs_ridge.to_csv('pred.csv')

