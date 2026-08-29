import pandas as pd

FILE_PATH = "data/raw/transactions.csv"

df = pd.read_csv(FILE_PATH)

print("\n===== DATASET OVERVIEW =====")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")

print("\n===== COLUMNS =====")
for column in df.columns:
    print(column)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== FRAUD DISTRIBUTION =====")
print(df["is_fraud"].value_counts())

print("\n===== FRAUD PERCENTAGE =====")
print(df["is_fraud"].value_counts(normalize=True) * 100)

print("\n===== FRAUD TYPES =====")
print(df["fraud_type"].value_counts())

print("\n===== TRANSACTION AMOUNT =====")
print(df["amount"].describe())

print("\n===== SAMPLE TRANSACTIONS =====")
print(df.head(10).to_string(index=False))

print("\n===== FRAUD TRANSACTIONS SAMPLE =====")
fraud_df = df[df["is_fraud"] == 1]
print(fraud_df.head(10).to_string(index=False))