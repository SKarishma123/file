import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
df = pd.read_csv('D:\Karishma\Python\Python_practice\sales_data_sample.csv', encoding = "latin-1")
numeric_columns = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_columns]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)
inertia = []
for n_clusters in range(1, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,n_init=10)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)
correlation_matrix = df_numeric.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)
plt.figure(figsize=(10, 8))
sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df['Cluster'], palette='viridis', s=50)
plt.title('Clusters Visualized using PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
#----------------------------------------------------------
#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
#from sklearn.datasets import make_blobs
#
## Generate sample sales data
#np.random.seed(42)
#data, _ = make_blobs(n_samples=300, centers=4, cluster_std=1.0, random_state=42)
#
## Create a DataFrame with the generated data
#sales_df = pd.DataFrame(data, columns=['Sales', 'Profit'])
#
## Visualize the data
#plt.scatter(sales_df['Sales'], sales_df['Profit'], s=50)
#plt.xlabel('Sales')
#plt.ylabel('Profit')
#plt.title('Sample Sales Data')
#plt.show()
#
## Perform k-means clustering
#k = 4  # Number of clusters
#kmeans = KMeans(n_clusters=k, random_state=42)
#sales_df['Cluster'] = kmeans.fit_predict(sales_df[['Sales', 'Profit']])
#
## Visualize the clustered data
#colors = ['red', 'green', 'blue', 'purple']
#for i in range(k):
#    cluster_data = sales_df[sales_df['Cluster'] == i]
#    plt.scatter(cluster_data['Sales'], cluster_data['Profit'], s=50, label=f'Cluster {i}', c=colors[i])
#
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='black', label='Centroids')
#plt.xlabel('Sales')
#plt.ylabel('Profit')
#plt.title('K-means Clustering of Sales Data')
#plt.legend()
#plt.show()
#
## Display cluster centers
#cluster_centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=['Sales', 'Profit'])
#print("Cluster Centers:")
#print(cluster_centers_df)










































