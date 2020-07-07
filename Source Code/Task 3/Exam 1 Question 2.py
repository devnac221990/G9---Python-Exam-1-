import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")
dataset = pd.read_csv('Customers.csv')
gender = dataset['Gender']
print(dataset.head())

X = dataset.iloc[:, [3, 4]]

print('X Shape (rows, col): ', X.shape)

sns.FacetGrid(dataset, hue="Spending Score (1-100)", size=5).map(plt.scatter, "Annual Income (k$)",
                                                                 "Spending Score (1-100)").add_legend()
plt.show()

from sklearn.cluster import KMeans
#Within Cluster Sum of Squares
wcss = []
# elbow method to know the number of clusters
# this loop will fit the k-means algorithm to our data and
# second we will compute the within cluster sum of squares and #appended to our wcss list.
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    # i above is between 1-15 numbers. init parameter is the random #initialization method
    # we select kmeans++ method. max_iter parameter the maximum number of iterations there can be to
    # find the final clusters when the K-meands algorithm is running. we #enter the default value of 300
    # the next parameter is n_init which is the number of times the #K_means algorithm will be run with
    # different initial centroid.
    # kmeans algorithm fits to the X dataset
    kmeans.fit(X)
    # kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
    wcss.append(kmeans.inertia_)

# kmeans inertia_ attribute is:  Sum of squared distances of samples #to their closest cluster center.
# 4.Plot the elbow graph
plt.plot(range(1, 16), wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

#5 According to the Elbow graph we deterrmine the clusters number as #Applying k-means algorithm to the X dataset.

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)

# 6 predict the cluster for each data point
# We are going to use the fit predict method that returns for each
# observation which cluster it belongs to. The cluster to which #client belongs and it will return this cluster numbers into a
# single vector that is  called y K-means
y_kmeans = kmeans.fit_predict(X)

# 7 we are going to calculate the score using silhouette method by importing metrics model from sklearn package
from sklearn import metrics
score = metrics.silhouette_score(X, y_kmeans)
print('Silhouette Score: ', score)

le = LabelEncoder()
le.fit_transform(gender)
dataset.loc[:, 'Gender'] = le.transform(gender)
scaler = StandardScaler()
scaler.fit(dataset)
data_scaled = scaler.transform(dataset)
data_scaled[0:3]

X = data_scaled[:, [3, 4]]
# use the fit predict method that returns for each #observation which cluster it belongs toThe cluster.
y_kmeans = kmeans.fit_predict(X)

# 6 Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red',label='BuyingGroup 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='BuyingGroup 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='BuyingGroup 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='BuyingGroup 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='BuyingGroup 5')

# Plot the centroid. This time we're going to use the cluster centres  #attribute that returns here the coordinates of the centroid.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='purple', marker='*',label='Centroids')
plt.title('Customers Buying Groups')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()

