#importing the libraries.
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd


#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/alldata/Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

#Splitting into train and test dataset.
"""from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =0 )
"""
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y.reshape(-1,1)).ravel()


#Fitting SVR to the dataset
#Create your regressor here
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X,Y)

#Predicting a new result
y_pred = regressor.predict([[6.5]])
y_pred = sc_Y.inverse_transform([y_pred])
#Visualizing the regression results
mp.scatter(X,Y,color="black") 
mp.plot(X, regressor.predict(X),color="blue")
mp.title("Position v/s salaries")
mp.xlabel("Position")
mp.ylabel("salaries")
mp.show()
