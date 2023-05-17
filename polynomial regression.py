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
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

#Fittiing Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Fitting Polynominal Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
## Now the matrix is created,we'll fit a linear model
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

#Visualizing the linear regression result
mp.scatter(X,Y,color='red')
mp.plot(X,lin_reg.predict(X), color = "blue")
mp.title("Position V/s Salaries")
mp.xlabel("Position")
mp.ylabel("Salaries")
mp.show()

#Visualizing the Polynomial regression result
X_grid = np.arange(min(X),max(X),0.1 )
X_grid = X_grid.reshape((len(X_grid),1))
mp.scatter(X,Y,color="black")
mp.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)),color="blue")
mp.title("Position v/s salaries")
mp.xlabel("Position")
mp.ylabel("salaries")
mp.show()

#Predicting for new X: linear model
lin_reg.predict(np.array([7]).reshape(-1, 1))
#Predicting for new X: polynomial model
lin_reg_2.predict(poly_reg.fit_transform(np.array([6.5]).reshape(-1,1)))