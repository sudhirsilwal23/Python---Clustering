#################################################################################
#####################  Hierarchical Clustering - Python  ########################
#################################################################################

#---------------------------------------------------------------------------------
# Step : 1 Importing the libraries
#---------------------------------------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#---------------------------------------------------------------------------------
# Step : 2 Data Preprocessing
#--------------------------------------------------------------------------------
         # Importing the dataset
dataset = pd.read_csv('Shopping_mall_card_subscribers.csv')


#--------------------------------------------------------------------------------
# Step : 3 Data modelling
#--------------------------------------------------------------------------------
        #3(a) Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(Var_Independent, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

       #3(b) Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
Var_D_hc = hc.fit_predict(Var_Independent)

#--------------------------------------------------------------------------------
# Step : 4 Data Visualising 
#--------------------------------------------------------------------------------
         #Visualising the clusters
plt.scatter(Var_Independent[Var_D_hc == 0, 0], Var_Independent[Var_D_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(Var_Independent[Var_D_hc == 1, 0], Var_Independent[Var_D_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(Var_Independent[Var_D_hc == 2, 0], Var_Independent[Var_D_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(Var_Independent[Var_D_hc == 3, 0], Var_Independent[Var_D_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(Var_Independent[Var_D_hc == 4, 0], Var_Independent[Var_D_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.Xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
