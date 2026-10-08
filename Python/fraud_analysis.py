import pandas as pd

# ==============================
# STEP 1: LOAD DATASET
# ==============================

print("Reading dataset...")

df = pd.read_csv(
    "credit_card_transactions.csv",
    dtype={
        "cc_num": "string",
        "trans_num": "string",
        "merchant": "string",
        "category": "string",
        "first": "string",
        "last": "string",
        "gender": "string",
        "street": "string",
        "city": "string",
        "state": "string",
        "job": "string"
    }
)

print("Dataset loaded successfully!")


# ==============================
# STEP 2: CLEAN DATA
# ==============================

# Remove unnecessary index column
df = df.drop(columns=['Unnamed: 0'])

# Convert transaction date
df['trans_date_trans_time'] = pd.to_datetime(
    df['trans_date_trans_time']
)

# Convert date of birth
df['dob'] = pd.to_datetime(df['dob'])

# Check duplicate rows
print("\nDuplicate rows:")
print(df.duplicated().sum())

# Check data types
print("\nData types:")
print(df.dtypes)


# ==============================
# STEP 3: CREATE NEW COLUMNS
# ==============================

# Transaction date
df['trans_date'] = df['trans_date_trans_time'].dt.date

# Transaction hour
df['trans_hour'] = df['trans_date_trans_time'].dt.hour

# Day of week
df['day_of_week'] = df['trans_date_trans_time'].dt.day_name()

# Month
df['month'] = df['trans_date_trans_time'].dt.month_name()

# Customer age
df['age'] = (
    (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25
).astype(int)


# ==============================
# CHECK RESULT
# ==============================

print("\nNew columns created successfully!")

print(
    df[
        ['trans_date',
         'trans_hour',
         'day_of_week',
         'month',
         'age']
    ].head()
)

print("\nFinal shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())

# ==============================
# STEP 4: BASIC FRAUD ANALYSIS
# ==============================

# Total transactions
total_transactions = len(df)

# Fraud transactions
fraud_transactions = df['is_fraud'].sum()

# Genuine transactions
genuine_transactions = (df['is_fraud'] == 0).sum()

# Fraud rate
fraud_rate = (fraud_transactions / total_transactions) * 100

# Total transaction amount
total_amount = df['amt'].sum()

# Fraud transaction amount
fraud_amount = df.loc[df['is_fraud'] == 1, 'amt'].sum()

print("\n===== FRAUD ANALYSIS =====")

print("Total Transactions:", total_transactions)
print("Genuine Transactions:", genuine_transactions)
print("Fraud Transactions:", fraud_transactions)

print("Fraud Rate:", round(fraud_rate, 2), "%")

print("Total Transaction Amount:", round(total_amount, 2))
print("Fraud Transaction Amount:", round(fraud_amount, 2))

# ==============================
# STEP 5: FRAUD PATTERN ANALYSIS
# ==============================

# 1. Fraud by Category
fraud_category = (
    df[df['is_fraud'] == 1]
    .groupby('category')
    .size()
    .sort_values(ascending=False)
)

print("\n===== FRAUD BY CATEGORY =====")
print(fraud_category)


# 2. Fraud by Merchant
fraud_merchant = (
    df[df['is_fraud'] == 1]
    .groupby('merchant')
    .size()
    .sort_values(ascending=False)
    .head(10)
)

print("\n===== TOP 10 MERCHANTS BY FRAUD =====")
print(fraud_merchant)


# 3. Fraud by State
fraud_state = (
    df[df['is_fraud'] == 1]
    .groupby('state')
    .size()
    .sort_values(ascending=False)
)

print("\n===== FRAUD BY STATE =====")
print(fraud_state)


# 4. Fraud by Hour
fraud_hour = (
    df[df['is_fraud'] == 1]
    .groupby('trans_hour')
    .size()
    .sort_values(ascending=False)
)

print("\n===== FRAUD BY HOUR =====")
print(fraud_hour)


# 5. Fraud by Gender
fraud_gender = (
    df[df['is_fraud'] == 1]
    .groupby('gender')
    .size()
    .sort_values(ascending=False)
)

print("\n===== FRAUD BY GENDER =====")
print(fraud_gender)


# 6. Fraud vs Genuine Amount Statistics
amount_analysis = (
    df.groupby('is_fraud')['amt']
    .agg(['count', 'sum', 'mean', 'max'])
)

print("\n===== AMOUNT ANALYSIS =====")
print(amount_analysis)

# ==============================
# STEP 6: FRAUD VISUALIZATIONS
# ==============================

import matplotlib.pyplot as plt


# 1. Fraud by Category
plt.figure(figsize=(10, 6))

fraud_category.sort_values().plot(kind='barh')

plt.title('Fraud Transactions by Category')
plt.xlabel('Number of Fraud Transactions')
plt.ylabel('Category')
plt.tight_layout()
plt.show()


# 2. Top 10 Merchants by Fraud
plt.figure(figsize=(10, 6))

fraud_merchant.sort_values().plot(kind='barh')

plt.title('Top 10 Merchants by Fraud Transactions')
plt.xlabel('Number of Fraud Transactions')
plt.ylabel('Merchant')
plt.tight_layout()
plt.show()


# 3. Fraud by Hour
plt.figure(figsize=(10, 6))

fraud_hour.sort_index().plot(kind='line', marker='o')

plt.title('Fraud Transactions by Hour')
plt.xlabel('Transaction Hour')
plt.ylabel('Number of Fraud Transactions')
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()
plt.show()


# 4. Fraud by State - Top 10
top_states = fraud_state.head(10).sort_values()

plt.figure(figsize=(10, 6))

top_states.plot(kind='barh')

plt.title('Top 10 States by Fraud Transactions')
plt.xlabel('Number of Fraud Transactions')
plt.ylabel('State')
plt.tight_layout()
plt.show()


# 5. Fraud vs Genuine Average Amount
average_amount = df.groupby('is_fraud')['amt'].mean()

plt.figure(figsize=(7, 5))

plt.bar(
    ['Genuine', 'Fraud'],
    [
        average_amount.get(0, 0),
        average_amount.get(1, 0)
    ]
)

plt.title('Average Transaction Amount: Fraud vs Genuine')
plt.xlabel('Transaction Type')
plt.ylabel('Average Amount')
plt.tight_layout()
plt.show()

# ==============================
# STEP 7: EXPORT CLEANED DATA
# ==============================

output_file = "credit_card_fraud_cleaned.csv"

df.to_csv(output_file, index=False)

print("\n===== EXPORT COMPLETE =====")
print("Cleaned dataset saved as:", output_file)
