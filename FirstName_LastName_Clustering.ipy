import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
transactions_df = pd.read_csv("Transactions.csv")
customers_df = pd.read_csv("Customers.csv")

# Merge datasets to prepare customer transaction features
merged_df = transactions_df.merge(customers_df, on="CustomerID", how="left")

# Aggregate transaction data per customer
customer_features = merged_df.groupby("CustomerID").agg(
    TotalTransactions=("TransactionID", "count"),
    TotalQuantity=("Quantity", "sum"),
    TotalSpend=("TotalValue", "sum"),
    AvgSpendPerTransaction=("TotalValue", "mean"),
).reset_index()

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(customer_features.iloc[:, 1:])

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
customer_features["Cluster"] = kmeans.fit_predict(scaled_features)

# Compute Davies-Bouldin Index
db_index = davies_bouldin_score(scaled_features, customer_features["Cluster"])
print(f"Davies-Bouldin Index: {db_index}")

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=customer_features["TotalSpend"],
    y=customer_features["TotalTransactions"],
    hue=customer_features["Cluster"],
    palette="viridis",
    alpha=0.7
)
plt.xlabel("Total Spend")
plt.ylabel("Total Transactions")
plt.title("Customer Clusters based on Spend and Transactions")
plt.legend(title="Cluster")
plt.show()
