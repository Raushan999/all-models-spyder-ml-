# Simple Linear Regression
##importing the libraries.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


##importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/alldata/Salary_Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

##Splitting into train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state =0 )

##Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
 
##Fitting simple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
##predicting the test set results
Y_pred = regressor.predict(X_test)

## Visualizing the training set results
plt.scatter(X_train,Y_train , color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title("Salary vs Experience(Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

##Visualizing the test set results
plt.scatter(X_test, Y_test)
plt.plot(X_train,regressor.predict(X_train), color = "black")
plt.title("salary vs experience(test data)")
plt.xlabel("yoe")
plt.ylabel('salary')
plt.show()

from sklearn.metrics import mean_squared_error
# Assuming you have already defined Y and Y_pred

accuracy = mean_squared_error(Y_test, Y_pred)
print("Accuracy:", accuracy)
#lesser is the mean sq error better is the model.

