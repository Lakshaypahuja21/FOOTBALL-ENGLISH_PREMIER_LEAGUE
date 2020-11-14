# -*- coding: utf-8 -*-
"""Regression Models

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1q-oKcGVvb6hu8t00Zs2qjgUli0WFrT_u

# lib
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
# %matplotlib inline

"""# dataset"""

football= pd.read_csv('football_CLEAN.csv')
football.head()

football.drop(['name','club','position','nationality'],axis=1,inplace=True)

football.head()

football['FPL_SEL'] = pd.to_numeric(football['fpl_sel'].str[:-1])
football.head()

football.drop(['fpl_sel'],axis=1,inplace=True)
football.head()

X= football.drop('market_value',axis=1)
Y= football['market_value']

from sklearn.model_selection  import train_test_split

"""# train test split"""

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state=101)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

football.isnull().sum()

"""# LINEAR REGRESSION"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

from sklearn import metrics

pred = regressor.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))

print(RMSE)

"""# LASSO REGRESSION"""

from sklearn.linear_model import Lasso

reg = Lasso(alpha=1.0)
reg.fit(X_train, Y_train)

pred= reg.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))

print(RMSE)

"""# RIDGE REGRESSION"""

from sklearn.linear_model import Ridge

clf = Ridge(alpha=1.0)
clf.fit(X_train, Y_train)

pred= clf.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))

print(RMSE)

"""# NEAREST NEIGHBOUR REGRESSOR"""

from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=5)
neigh.fit(X_train, Y_train)

pred= neigh.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))
print(RMSE)

"""# SUPPORT VECTOR REGRESSOR"""
from sklearn.model_selection import cross_val_score
from sklearn import svm

from sklearn.svm import SVR
sregressor = SVR(kernel = 'rbf')
sregressor.fit(X_train, Y_train)

pred= sregressor.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))
print(RMSE)

"""# TREE REGRESSOR"""

from sklearn.tree import DecisionTreeRegressor
tregressor = DecisionTreeRegressor(random_state=0)
tregressor.fit(X_train, Y_train)

pred= tregressor.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))
print(RMSE)

"""# RANDOM FOREST REGRESSOR"""

from sklearn.ensemble import RandomForestRegressor
rregressor = RandomForestRegressor(n_estimators = 100, random_state = 0)
rregressor.fit(X_train, Y_train)

pred= rregressor.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))
print(RMSE)

"""# GRADIENT BOOSTING REGRESSOR"""

from sklearn.ensemble import GradientBoostingRegressor
REGGB = GradientBoostingRegressor(random_state=0)
REGGB.fit(X_train, Y_train)

pred= REGGB.predict(X_test)

print(pred)

RMSE = np.sqrt(metrics.mean_squared_error(Y_test, pred))
print(RMSE)



pickle.dump(REGGB, open('fmodel.pkl','wb'))

# fmodel = pickle.load(open('fmodel.pkl','rb'))

# print(fmodel.predict())