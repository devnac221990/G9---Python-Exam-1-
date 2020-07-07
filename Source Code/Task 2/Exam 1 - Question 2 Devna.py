import numpy as np
import pandas as pd


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from pandas import plotting
import random
random.seed(50)
cc = pd.read_csv("C:/Users/Devna Chaturvedi/Desktop/Python Exam 1/Question 1/Customers.csv")
cc.head()
print(cc.head())
cc.shape
print(cc.shape)
cc.info()
cc.describe()
print(cc.describe())
# Null values
cc.isnull().sum().sort_values(ascending=False)
print(cc.isnull().sum().sort_values(ascending=False))
# to drop the customer id
cc = cc.drop('CustomerID', axis=1)
cc.head(2)
print(cc.head(2))
new_cols = ['Gender', 'Age', 'AnnualIncome','SpendingScore']

cc.columns = new_cols

cc.head(3)
print(cc.head(3))
train_X, test_X = train_test_split(cc, test_size=0.25, random_state=50)

print(len(train_X), "train +", len(test_X), "test")
# copy the data for training the dataset
cc2 = train_X.copy()
# to fit and transform the Gender attribute into numeric
le = LabelEncoder()
le.fit(cc2.Gender)
print(le.fit(cc2.Gender))
# 0 is Female, 1 is Male
le.classes_

print(le.classes_)
#update cc2 with transformed values of gender
cc2.loc[:,'Gender'] = le.transform(cc2.Gender)
# Create scaler: scaler
scaler = StandardScaler()
scaler.fit(cc2)
# transform
data_scaled = scaler.transform(cc2)
data_scaled[0:3]
pca = PCA()

# PCA
pca.fit(data_scaled)
# PCA features
features = range(pca.n_components_)
features
# PCA transformed data
data_pca = pca.transform(data_scaled)
data_pca.shape

pca.explained_variance_ratio_


pca2 = PCA(n_components=2, svd_solver='full')

# fit PCA
pca2.fit(data_scaled)

# PCA transformed data
data_pca2 = pca2.transform(data_scaled)
data_pca2.shape
xs = data_pca2[:,0]
ys = data_pca2[:,1]

plt.scatter(ys, xs)


plt.grid(False)
plt.title('Scatter Plot of Customers data')
plt.xlabel('PCA-01')
plt.ylabel('PCA-02')

plt.show()


k=5
kmeans = KMeans(n_clusters=k, init = 'k-means++',random_state = 50)
# Build pipeline
pipeline = make_pipeline(scaler, pca2, kmeans)

# to fit the model into the dataset
model_fit = pipeline.fit(cc2)
model_fit

labels = model_fit.predict(cc2)
labels
# lets add the clusters to the dataset
train_X['Clusters'] = labels

train_X.groupby('Clusters').count()
print(train_X.groupby('Clusters').count())


xs = data_pca2[:,0]
ys = data_pca2[:,1]

plt.scatter(ys, xs,c=labels)


plt.grid(False)
plt.title('Scatter Plot of Customers data')
plt.xlabel('PCA-01')
plt.ylabel('PCA-02')

plt.show()

model_fit[2].inertia_
print(model_fit[2].inertia_)
# WCSS

ks = range(1, 10)
wcss = []
samples = data_pca2

for i in ks:
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 50)
    kmeans.fit(samples)

    wcss.append(kmeans.inertia_)



def getInertia2(X,kmeans):

    inertia2 = 0
    for K in range(len(X)):
        L = min(1,len(kmeans.cluster_centers_)-1) # this is just for the case where there is only 1 cluster at all
        dist_to_center = sorted([np.linalg.norm(X[k] - z)**2 for z in kmeans.cluster_centers_])[L]
        inertia2 = inertia2 + dist_to_center
    return inertia2

wcss = []
inertias_2 = []
silhouette_avgs = []

ks = range(1, 10)
samples = data_pca2

for t in ks:
    kmeans = KMeans(n_clusters = t, init = 'k-means++', random_state = 50)
    kmeans.fit(samples)
    wcss.append(kmeans.inertia_)
    inertias_2.append(getInertia2(samples,kmeans))
    if t>1:
        silhouette_avgs.append(silhouette_score(samples, kmeans.labels_))

silhouette_avgs
print(silhouette_avgs)


plt.figure(figsize=(10,5))
sns.lineplot(ks, wcss,marker='o',color='blue')
plt.title('elbow technique')
plt.xlabel('total clusters')
plt.ylabel('WCSS')
plt.show()
