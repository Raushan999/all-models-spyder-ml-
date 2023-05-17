#importing the libraries.
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd



#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/alldata/50_Startups.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values


#Encoding Categorical data
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = ct.fit_transform(X)
# rounding to the nearest integer
X= X.astype(float)
X = np.rint(X)
#Taking care of dummy variable trap.
X = X[:, 1:]


#Splitting into train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =0 )


#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fitting Multiple Linear Regression model on Training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting result with the test dataset
Y_pred = regressor.predict(X_test)

#Building the optimal model using Backward Elimination
import statsmodels.api as sm
X = np.append(arr=np.ones((50,1), dtype = int), values=X,axis =1)
X_opt = X[:, [0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
##second iteration
X_opt = X[:, [0,1,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
##third iteration
X_opt = X[:, [0,3,4,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

##fourth iteration
X_opt = X[:, [0,3,5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()

