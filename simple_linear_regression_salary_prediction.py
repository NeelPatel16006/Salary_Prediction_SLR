# -*- coding: utf-8 -*-
"""Simple_Linear_Regression_Salary_Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SCQIFzrVnEllefscuR83oS1MzQYCRUBv

# Import Libraries
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import pickle

"""# Read the data"""

df=pd.read_csv('Salary_Data (1).csv')
df.head()

df.info()

df.describe()

"""# Data Visulization"""

plt.figure(figsize=(8,8))
sns.distplot(df['Salary'])
plt.show()

plt.figure(figsize=(8,8))
sns.distplot(df['YearsExperience'])
plt.show()

plt.figure(figsize=(8,8))
sns.scatterplot(x='Salary',y='YearsExperience',data=df)
plt.show()

sns.pairplot(df)
plt.show()

"""# Data Preparation"""

X=df['YearsExperience']
y=df['Salary']

print(X.shape)
print(y.shape)

X=X.values.reshape(-1,1)
y=y.values.reshape(-1,1)

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)

print('shape of X_train: ',X_train.shape)
print('shape of X_test: ',X_test.shape)
print('shape of y_train: ',y_train.shape)
print('shape of y_test: ',y_test.shape)

"""# Model Building"""

lm=LinearRegression()
lm.fit(X_train,y_train)

y_pred= lm.predict(X_test)

print('model coeficient is :',lm.coef_)
print('model intercept is : ',lm.intercept_)

"""# Check Model Performance"""

print('RMSE is : ', mean_squared_error(y_test,y_pred,squared=False))

print('R2 score is :',r2_score(y_test,y_pred))

plt.figure(figsize=(10,7))
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred,c='black')
plt.title('Relationship between Salary  and Year of Experience')
plt.xlabel('Year OF Experience')
plt.ylabel('Salary')
plt.legend(['Salary','Year of Experience'],loc='upper left')
plt.show()

residual = y_test-y_pred
print(residual)

"""# Checing overfitting and underfitting"""

print('Training score is: ',lm.score(X_train,y_train))
print('Testing score is: ',lm.score(X_test,y_test))

"""# Check assumptions

normality of residuals
"""

sns.distplot(residual)

"""Homoscedacity"""

fig,ax=plt.subplots(figsize=(6,2.5))
_ = ax.scatter(y_pred,residual)

"""No auto-correlation of residuals"""

#acf=smt.graphics.plot_acf(residual,alpha=0.05)
#acf.show()

# saving  model to disk
filename = 'finalized_model.pkl'
pickle.dump(lm, open(filename, 'wb'))

# loading model to compare to result

loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model.predict([[2]]))