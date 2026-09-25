import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as  sns
import warnings 
warnings.filterwarnings("ignore")

df = pd.read_csv("titanic.csv")
# print(df.head())
columns = df.columns.to_list()
# print(columns)
# df_info = df.info()
df.drop(["embark_town", "deck", "alive", "class", "who", "adult_male"], axis=1, inplace=True)
# print(df)
df["age"] = df["age"].fillna(df['age'].mean(), inplace=True) #filling missing values(156 values) with mean 
df.dropna(subset=["embarked"], inplace=True) #dropping missing rows bcoz only 3 rows are missing values so values 892 becomes 889
# df.info()

# Label Encoding
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()
df['sex'] = le.fit_transform(df["sex"])
df['embarked'] = le.fit_transform(df["embarked"])  #s=2 c=0 q=1
df = df.astype(int)
# print(df.head())

X = df.drop("survived", axis=1)
y = df['survived']

# ===== Feature Scaling =====

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
# print(X_train_scaled)

X_test_scaled = scaler.fit_transform(X_test)
# print(X_test_scaled)

# ===== IMPLEMENTING KNN Algo ======

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled,y_train)
y_predct = model.predict(X_test_scaled)
# print(y_predct)
# print(y_test)

# ===== Model Evaluation ==========

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc_score = accuracy_score(y_test, y_predct)
print("Accuracy Score:", acc_score)

cnf_matrix = confusion_matrix(y_test,y_predct)
print(cnf_matrix)
clf_report = classification_report(y_test,y_predct)
print(clf_report) 