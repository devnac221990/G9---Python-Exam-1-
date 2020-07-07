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
# Ues elbow method with range 1-16 to find the optimal number of k
for i in range(1, 16):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)


# Visualize elbow
plt.plot(range(1, 16), wcss)
plt.title('The Elbow Method Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# from the elbow figure we can determine the best k is 5.

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=0)


# We are using the fit predict method to find y-predict

y_kmeans = kmeans.fit_predict(X)

# silhouette method to find the accuracy score
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

y_kmeans = kmeans.fit_predict(X)

# Visualising the Buying Groups
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red',label='BuyingGroup 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='blue', label='BuyingGroup 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='green', label='BuyingGroup 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='BuyingGroup 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='BuyingGroup 5')

# add centroid to graph.
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='purple', marker='*',label='Centroids')
plt.title('Customers Buying Groups')
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score(1-100)')
plt.show()

