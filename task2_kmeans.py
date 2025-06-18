# Importing necessary libraries for data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

# Load dataset
data = pd.read_csv("dataset/customers.csv")

# Displaying the first few rows of the dataset
print("Sample Data:")
print(data.head())

# Displaying dataset information including column types and non-null counts
print("\nDataset Info:")
print(data.info())

# Checking for missing values in the dataset
print("\nMissing Values:")
print(data.isnull().sum())

# Generating summary statistics for numerical features
print("\nSummary Statistics:")
print(data.describe())

# Visualizing distributions of 'Age' and 'Annual Income'
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
sns.histplot(data['Age'], bins=20, kde=True, color='skyblue')
plt.title("Age Distribution")

plt.subplot(1, 2, 2)
sns.histplot(data['Annual Income (k$)'], bins=20, kde=True, color='orange')
plt.title("Annual Income Distribution")

plt.tight_layout()
plt.show()

# Selecting relevant features for clustering
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]

# Finding the optimal number of clusters using the Elbow Method
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--', color='purple')
plt.title('Elbow Method to Determine Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid(True)
plt.show()

# Training the KMeans model with the optimal number of clusters
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)

# Visualizing the formed clusters
plt.figure(figsize=(8, 5))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(X.values[y_kmeans == i, 0], X.values[y_kmeans == i, 1],
                s=100, c=colors[i], label=f'Cluster {i+1}')

# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=300, c='yellow', label='Centroids', marker='*')

plt.title('Customer Segmentation using K-Means Clustering')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid(True)
plt.show()
