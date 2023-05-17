from pyECLAT import ECLAT

import pandas as pd
#importing the dataset
dataset = pd.read_csv("C:/Users/HP/Downloads/alldata/Market_Basket_Optimisation.csv", header = None)

# Applying ECLAT algorithm
eclat_instance = ECLAT(data=dataset, verbose=True)
frequent_itemsets = eclat_instance.fit()

# Displaying the frequent itemsets
for itemset, support in frequent_itemsets.items():
    print(itemset, support)
