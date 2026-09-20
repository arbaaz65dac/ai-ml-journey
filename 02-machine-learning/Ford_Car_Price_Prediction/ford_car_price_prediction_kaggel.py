import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv("ford.csv")

# ==== EDA ====
df_head = df.head()
# print("DataFrame Head:\n",df_head)

df_shape = df.shape
# print("DF Shape:\n",df_shape)

# df_info = df.info()
# print("DF Info:\n", df_info)

df_desc = df.describe()
# print("DF Desc:",df_desc)

# print(df.isnull().sum())  #Check null

# EDA for Price

sns.histplot(df['price'], bins= 50, kde=True)
# plt.show()

# Correlation Check

df_correlation = df.corr(numeric_only=True)
# print(df_correlation)
sns.heatmap(df_correlation, annot= True)
# plt.show()

# price vs year
sns.boxplot(data=df, x='year', y='price')
plt.xticks(rotation = 90)
# plt.show()

# Mileage vs Price
sns.scatterplot(data=df, x='mileage', y='price')
# plt.show()

# price vs engineSize
sns.boxplot(data=df, x='engineSize', y='price')
# plt.show()

# price vs transmission
sns.boxplot(data=df, x='transmission', y='price')
# plt.show()

# price vs fuelType
sns.boxplot(data=df, x='fuelType', y='price')
# plt.show()

# price vs model
sns.boxplot(data=df, x='model', y='price')
plt.xticks(rotation = 90)
# plt.show()

# price vs mpg
sns.boxplot(data=df, x='model', y='price')
plt.xticks(rotation = 90)
# plt.show()

# price vs tax
sns.boxplot(data=df, x='model', y='price')
plt.xticks(rotation = 90)
# plt.show()

# ===== Separate Features and Target =====

X = df.drop(columns=['price'])
y = df['price']


# ===== Data Preprocessing =====

# Encoding categorical columns
X_one_encoded = pd.get_dummies(
    X,
    columns=['model', 'transmission', 'fuelType'],
    drop_first=True
)

X_one_encoded = X_one_encoded.astype(int)

# print(X_one_encoded)

# ==== Label Encoding =====
from sklearn.preprocessing import LabelEncoder

columns = ['model', 'transmission', 'fuelType'] 

X_lable_encoding = X.copy()
label_encoders = {}
for i in columns:
    le = LabelEncoder()
    X_lable_encoding[i] = le.fit_transform(X_lable_encoding[i].astype(str)) # convert to string in case of null
    label_encoders[i] = le
# print(X_lable_encoding)

# Standard Scaler for scaling (in between -3 and 3)
from sklearn.preprocessing import StandardScaler

numerical_columns = ['year','mileage','tax','engineSize']
scaler = StandardScaler()

X_one_encoded[numerical_columns] = scaler.fit_transform(X_one_encoded[numerical_columns]) 
# print(X_one_encoded)

# print(X_lable_encoding.columns.to_list())
X_lable_encoding[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']] = scaler.fit_transform(
    X_lable_encoding[['model', 'year', 'transmission', 'mileage', 'fuelType', 'tax', 'mpg', 'engineSize']])
# print(X_lable_encoding)

# ====== Model Creation =========

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Model with One Hot Encoding 
X_train, X_test, y_train, y_test = train_test_split(X_one_encoded, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train) 
# print(model)

y_predictions = model.predict(X_test)
# print(y_predictions)
# print(y_test)

# ========== Model Evaluation ==========
# ========== Comparing Actual Value with predicted Value using R^2 ==========

r2 = r2_score(y_test, y_predictions) # R^2 
print("One Hot Encoded R^2:",r2)

# Adjusted R^2
n = X_test.shape[0]  #No. of rows
p = X_test.shape[1]  #No. of Columns

adjusted_r2 = 1 - ((1 - r2) * (n-1)/ (n - p - 1))
print("One Hot Encoded Adjusted R^2:", adjusted_r2)

# ===========================================
#Model with Label Encoding
X_train, X_test, y_train, y_test = train_test_split(X_lable_encoding, y, test_size=0.33, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train) 
# print(model)

y_predictions = model.predict(X_test)
# print(y_predictions)
# print(y_test)

# ========== Model Evaluation ==========
# ========== Comparing Actual Value with predicted Value using R^2 ==========

r2 = r2_score(y_test, y_predictions) # R^2 
print("Label Encoded R^2:",r2)

# Adjusted R^2
n = X_test.shape[0]  #No. of rows
p = X_test.shape[1]  #No. of Columns

adjusted_r2 = 1 - ((1 - r2) * (n-1)/ (n - p - 1))
print("Label Encoded Adjusted R^2:", adjusted_r2)
