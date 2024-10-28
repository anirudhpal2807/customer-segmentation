# customer-segmentation
Customer segmentation using K-means clustering is a popular approach for identifying distinct groups within a customer base based on common attributes. By grouping customers into clusters, businesses can tailor marketing, sales strategies, and product development efforts to suit the needs of each segment.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
Customer_data = pd.read_csv('/Mall_Customers.csv')
Customer_data.head()
Customer_data.shape
#getting information of data
Customer_data.info()
# see any missing value
Customer_data.isnull().sum()
X=Customer_data.iloc[:,[3,4]].values
sns.set
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Point Graph')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
#
y_kmeans=kmeans.fit_predict(X)
print(y_kmeans)
#ploting all the cluster and respective their centroid
plt.figure(figsize=(10,10))
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=10,c='red',label='cluster 1')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=10,c='blue',label='cluster 2')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=10,c='green',label='cluster 3')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=10,c='black',label='cluster 4')
plt.scatter(X[y_kmeans==4,0],X[y_kmeans==4,1],s=10,c='yellow',label='cluster 5')
#plt.scatter(X[y_kmeans==5,0],X[y_kmeans==5,1],s=10,c='pink',label='cluster 6')

 #plotting the centroid
 # s represent the size of dot
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=50,c='cyan',label='centroid')
plt.title('cluster of customer')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()
cluster_3_data = Customer_data[y_kmeans == 2]
print(cluster_3_data)
# these customer have high number of annual income but not spent much money
