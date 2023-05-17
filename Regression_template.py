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

#Fitting Regression Model to the dataset
#Create your regressor here

#Predicting a new result
Y_pred = regressor.predict(np.array(6.5).reshape(-1,1))

#Visualizing the regression results
mp.scatter(X,Y,color="black")
mp.plot(X, regressor.predict(X),color="blue")
mp.title("Position v/s salaries")
mp.xlabel("Position")
mp.ylabel("salaries")
mp.show()

#Visualizing the regression result (for higher resolution and smooth curve)
X_grid = np.arange(min(X),max(X),0.1 )
X_grid = X_grid.reshape((len(X_grid),1))
mp.scatter(X,Y,color="black")
mp.plot(X_grid, regressor.predict(X_grid),color="blue")
mp.title("Position v/s salaries")
mp.xlabel("Position")
mp.ylabel("salaries")
mp.show()

