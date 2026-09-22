import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")

#EDA (Exploratory Data Analysis)
df_shape = df.shape
print("Shape:", df_shape)

df_head = df.head()
print("Head:", df_head)

df_info = df.info()
print("Info:", df_info)

df_describe = df.describe()
print("Describe:", df_describe)

df_isnull = df.isnull().sum()
print("Missing values:\n", df_isnull)

numeric_columns = ["age", "bmi", "children", "charges"]
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

sns.countplot(x="children", data=df)
plt.title("Count of Children")
plt.xlabel("Number of Children")
plt.ylabel("Count")
plt.show()

sns.countplot(x="sex", data=df)
plt.title("Count of Sex")
plt.xlabel("Sex")
plt.ylabel("Count") 
plt.show()

sns.countplot(x="smoker", data=df)
plt.title("Count of Smoker")
plt.xlabel("Smoker")
plt.ylabel("Count")
plt.show()

# # How input features are related with output feature (charges)
for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=df[col])
    plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# ===============================================================================

# Data Cleaning and Preprocessing
df_cleaned = df.copy()

df_cleaned.drop_duplicates(inplace=True)
print("Shape after dropping duplicates:", df_cleaned.shape)

value_counts = df_cleaned["sex"].value_counts()  # Male,male,MALE etc
print("Value counts for 'sex':\n", value_counts)

# Label Encoding for categorical variables
df_cleaned["sex"] = df_cleaned["sex"].str.lower().map({"male": 0, "female": 1})
df_cleaned["smoker"] = df_cleaned["smoker"].str.lower().map({"no": 0, "yes": 1})
# print(df_cleaned.head())

# Renaming columns for better understanding
df_cleaned.rename(columns={"sex": "is_female", "smoker": "is_smoker"}, inplace=True)              
# print(df_cleaned.head())

# One-hot encoding for 'region' column
value_counts_region = df_cleaned["region"].value_counts()
print("Value counts for 'region':\n", value_counts_region)

df_cleaned = pd.get_dummies(df_cleaned, columns=["region"], drop_first=True)
df_cleaned = df_cleaned.astype(int)  # Convert all columns to integer type
print("Columns after one-hot encoding:\n", df_cleaned.head())

# =================================================================================

# Feature Engineering and Extraction
df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins=[0, 18.5, 24.9, 29.9, float('inf')],
    labels=['Underweight', 'Normal', 'Overweight', 'Obese']
)

# print(df_cleaned.head())

df_cleaned = pd.get_dummies(df_cleaned, columns=["bmi_category"], drop_first=True)
df_cleaned = df_cleaned.astype(int)  # Convert all columns to integer type
# print(df_cleaned.head())

# Feature Scaling =======

from sklearn.preprocessing import StandardScaler
cols = ['age', 'bmi', 'children']

scaler = StandardScaler()
df_cleaned[cols] = scaler.fit_transform(df_cleaned[cols])
# print(df_cleaned.head())

#Feature Extraction =======

from scipy.stats import pearsonr

# Pearson correlation Calculation 
selected_features = [
    'age', 'bmi', 'children', 'is_female',
    'is_smoker', 'region_northwest', 'region_southwest', 'region_southeast',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

# Calculate Pearson correlation for selected features with charges

correlations = {
    feature: pearsonr(
        df_cleaned[feature],
        df_cleaned["charges"]
    )[0]
    for feature in selected_features
}

# Create DataFrame
correlation_df = pd.DataFrame(
    list(correlations.items()),
    columns=["Feature", "Pearson Correlation"]
)

# Sort by correlation
correlation_df = correlation_df.sort_values(
    by="Pearson Correlation",
    ascending=False
)

# Display result
print(correlation_df.head())

cat_features = [
    'is_female','is_smoker',
    'region_northwest', 'region_southwest', 'region_southeast',
    'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

from scipy.stats import chi2_contingency

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)

chi2_results = {}

for col in cat_features:
    contingency = pd.crosstab(
        df_cleaned[col],
        df_cleaned['charges_bin']
    )

    chi2_stat, p_val, _, _ = chi2_contingency(contingency)

    decision = (
        'Reject Null (Keep Feature)'
        if p_val < alpha
        else 'Accept Null (Drop Feature)'
    )

    chi2_results[col] = {
        'chi2_statistic': chi2_stat,
        'p_value': p_val,
        'Decision': decision
    }

chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by = 'p_value')
# print(chi2_df)

final_df = df_cleaned[['age','is_female','bmi', 'children','is_smoker', 'charges','region_southeast', 'bmi_category_Obese']]
print("Final Data :")
print(final_df)

# =============== Model Selection ==============
# =============== Linear Regression Model ==============

from sklearn.model_selection import train_test_split

X = final_df.drop('charges', axis = 1)
y = final_df['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) #80% training 20% testing split

from sklearn.linear_model import LinearRegression  # Linear regression Creation

model = LinearRegression()

model.fit(X_train, y_train) 
# print(model)

y_predictions = model.predict(X_test)

# print(y_predictions)
# print(y_test)

# ========== Model Evaluation ==========
# ========== Comparing Actual Value with predicted Value  ==========

from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_predictions) # R^2 
print("R^2:",r2)

# Adjusted R^2
n = X_test.shape[0]  #No. of rows
p = X_test.shape[1]  #No. of Columns

adjusted_r2 = 1 - ((1 - r2) * (n-1)/ (n - p - 1))
print("Adjusted R^2:", adjusted_r2)

 



