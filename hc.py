#Hierarchial Clustering
#import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/alldata/Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values

#using the dendogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X,method='ward')) # ward method minimized the variance

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()
#optimal number of clusters = find the largest vertical distance without crossing any horizontal line
#then in that vert-distance count all the vertical lines available. here, we have 5.

#Fitting hierarchial clustering to the mall dataset
#(same algo as the normal clustering)
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_hc = hc.fit_predict(X)

#Visualizing the clusters.
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100,c='red',label='careful')
plt.scatter(X[y_hc==1,0],X[y_hc==1,1],s=100,c='blue',label='standard')
plt.scatter(X[y_hc==2,0],X[y_hc==2,1],s=100,c='green',label='target')
plt.scatter(X[y_hc==3,0],X[y_hc==3,1],s=100,c='cyan',label='careless')
plt.scatter(X[y_hc==4,0],X[y_hc==4,1],s=100,c='magenta',label='sensible')
plt.title('clusters of customers')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.show()

