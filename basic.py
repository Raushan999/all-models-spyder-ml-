#importing the libraries.
import numpy as np
import matplotlib.pyplot as mp
import pandas as pd

#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
X

#Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])
# create a column transformer to encode the first column
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# fit the transformer to the data and transform the data
X = ct.fit_transform(X)
# rounding to the nearest integer
X= X.astype(float)
X = np.rint(X)


#creating label encoder for Y
labelencoder_Y = LabelEncoder()
Y= labelencoder_Y.fit_transform(Y)

#Splitting into train and test dataset.
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state =0 )

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)



