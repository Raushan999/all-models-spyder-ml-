#importing the libraries.
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd


#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Splitting into train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =0 )

#Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""



