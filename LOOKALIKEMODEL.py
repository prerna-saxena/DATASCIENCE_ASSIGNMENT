import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
customers_df = pd.read_csv("Customers.csv")
products_df = pd.read_csv("Products.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Merge transactions with customer and product data
transactions_df = transactions_df.merge(customers_df, on="CustomerID")
transactions_df = transactions_df.merge(products_df, on="ProductID")

# Aggregate transaction history per customer
customer_summary = transactions_df.groupby("CustomerID").agg({
    "TotalValue": "sum",
    "Quantity": "sum",
    "Price": "mean"
}).reset_index()

# Normalize data for similarity computation
scaler = StandardScaler()
customer_features = scaler.fit_transform(customer_summary.iloc[:, 1:])

# Compute similarity scores
similarity_matrix = cosine_similarity(customer_features)

# Function to find similar customers
def find_similar_customers(customer_id, n=3):
    if customer_id not in customer_summary["CustomerID"].values:
        return "Customer ID not found."
    
    idx = customer_summary[customer_summary["CustomerID"] == customer_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similar_customers = [(customer_summary.iloc[i[0]]['CustomerID'], i[1]) for i in similarity_scores[1:n+1]]
    return similar_customers

# Example usage
customer_id = 101  # Replace with actual customer ID
similar_customers = find_similar_customers(customer_id)
print("Similar Customers:", similar_customers)
