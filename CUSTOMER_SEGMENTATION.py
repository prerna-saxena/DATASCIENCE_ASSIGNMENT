"""
Customer Segmentation using Clustering

This script performs customer segmentation by clustering customers based on their profile information 
and transaction history. The goal is to group customers with similar purchasing behaviors and demographics.

Steps:
1. Load and merge datasets (Customers, Transactions).
2. Aggregate transaction data per customer.
3. Encode categorical variables and normalize numerical features.
4. Apply a clustering algorithm (K-Means) to segment customers.
5. Determine the optimal number of clusters using the Elbow Method.
6. Visualize and analyze the resulting customer segments.
7. Save the clustered data for further analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Load datasets
customers_df = pd.read_csv("Customers.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Merge transaction data with customer data
transactions_df = transactions_df.merge(customers_df, on="CustomerID")

# Aggregate transaction history per customer
customer_summary = transactions_df.groupby("CustomerID").agg({
    "TotalValue": "sum",
    "Quantity": "sum"
}).reset_index()

# Merge aggregated data with customer profile
customer_data = customers_df.merge(customer_summary, on="CustomerID", how="left").fillna(0)

# Encode categorical variables (Region)
le = LabelEncoder()
customer_data["Region"] = le.fit_transform(customer_data["Region"])

# Select features for clustering
features = ["Region", "TotalValue", "Quantity"]
X = customer_data[features]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
inertia = []
k_values = range(2, 11)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--')
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()

# Apply K-Means with optimal clusters (manually set based on elbow method)
optimal_k = 4  # Change this based on the elbow curve result
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
customer_data["Cluster"] = kmeans.fit_predict(X_scaled)

# Visualize Clusters using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
customer_data["PCA1"] = X_pca[:, 0]
customer_data["PCA2"] = X_pca[:, 1]

plt.figure(figsize=(8, 6))
sns.scatterplot(x=customer_data["PCA1"], y=customer_data["PCA2"], hue=customer_data["Cluster"], palette="viridis")
plt.title("Customer Segmentation Clusters")
plt.show()

# Save segmented customers
target_path = "Customer_Segmentation.csv"
customer_data.to_csv(target_path, index=False)
print(f"Customer segmentation results saved to {target_path}")
