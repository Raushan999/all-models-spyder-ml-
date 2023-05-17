#Apriori 
#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/alldata/Market_Basket_Optimisation.csv", header = None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on the dataset
from apyori import apriori

rules = apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)

#visualizing the results
results = list(rules)
similar_items = []

for rule in results:
    itemset = [item for item in rule.items]
    support = rule.support
    confidence = rule.ordered_statistics[0].confidence
    lift = rule.ordered_statistics[0].lift
    similar_items.append((itemset, support, confidence, lift))












