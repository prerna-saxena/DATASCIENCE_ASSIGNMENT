"""
Lookalike Model Development

This script loads customer, product, and transaction data, processes it, and builds a similarity-based recommendation model. 
It identifies the top 3 most similar customers for a given user based on purchase history and spending behavior.

Steps:
1. Load and merge datasets (Customers, Products, Transactions).
2. Aggregate transaction data per customer.
3. Normalize customer features using StandardScaler.
4. Compute similarity scores using Cosine Similarity.
5. Generate Lookalike recommendations for the first 20 customers.
6. Save the results in 'Lookalike.csv'.
"""

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
        return []
    
    idx = customer_summary[customer_summary["CustomerID"] == customer_id].index[0]
    similarity_scores = list(enumerate(similarity_matrix[idx]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    similar_customers = [(customer_summary.iloc[i[0]]['CustomerID'], round(i[1], 4)) for i in similarity_scores[1:n+1]]
    return similar_customers

# Filter first 20 customers
first_20_customers = customers_df.iloc[:20]["CustomerID"].tolist()

# Create a lookalike mapping
lookalike_results = {cust_id: find_similar_customers(cust_id, n=3) for cust_id in first_20_customers}

# Convert to DataFrame and save as CSV
lookalike_df = pd.DataFrame(list(lookalike_results.items()), columns=["CustomerID", "Lookalikes"])
lookalike_df.to_csv("Lookalike.csv", index=False)

print("Lookalike.csv file has been created successfully!")
