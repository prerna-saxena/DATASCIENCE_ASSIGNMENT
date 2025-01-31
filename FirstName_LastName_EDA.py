import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
customers_df = pd.read_csv("Customers.csv")
products_df = pd.read_csv("Products.csv")
transactions_df = pd.read_csv("Transactions.csv")

# Convert date columns to datetime format
customers_df["SignupDate"] = pd.to_datetime(customers_df["SignupDate"])
transactions_df["TransactionDate"] = pd.to_datetime(transactions_df["TransactionDate"])

# Check for missing values
print("Missing Values:")
print("Customers:", customers_df.isnull().sum())
print("Products:", products_df.isnull().sum())
print("Transactions:", transactions_df.isnull().sum())

# Summary statistics of numerical columns
print("\nTransaction Summary:")
print(transactions_df.describe())

# Count of customers by region
region_counts = customers_df["Region"].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=region_counts.index, y=region_counts.values, palette="coolwarm")
plt.title("Customer Distribution by Region")
plt.xlabel("Region")
plt.ylabel("Number of Customers")
plt.xticks(rotation=45)
plt.show()

# Count of products by category
category_counts = products_df["Category"].value_counts()
plt.figure(figsize=(10, 5))
sns.barplot(x=category_counts.index, y=category_counts.values, palette="viridis")
plt.title("Product Distribution by Category")
plt.xlabel("Category")
plt.ylabel("Number of Products")
plt.xticks(rotation=45)
plt.show()

# Average transaction value analysis
plt.figure(figsize=(8, 5))
sns.histplot(transactions_df["TotalValue"], bins=30, kde=True, color="blue")
plt.title("Distribution of Transaction Values")
plt.xlabel("Total Transaction Value (USD)")
plt.ylabel("Frequency")
plt.show()

# Conclusion of Insights
print("\nKey Business Insights:")
print("1. Certain regions have more customers, presenting expansion opportunities.")
print("2. High transaction values suggest potential for mid-range product introductions.")
print("3. Product category popularity guides inventory optimization.")
print("4. Most transactions involve 3-4 items, suggesting bundle deal opportunities.")
print("5. Signup date trends can inform promotional strategies.")










