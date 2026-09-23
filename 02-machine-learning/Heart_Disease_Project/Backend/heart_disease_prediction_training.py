import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

#EDA & Data Cleaning  ==========

df = pd.read_csv('heart.csv')
df_head = df.head()
# print("Data Head :", df_head)

df_shape = df.shape
# print("shape:",df_shape)

df_columns = df.columns.to_list()
# print("Columns:", df_columns)

# df_info = df.info()
# print("Info:\n", df_info)

df_describe = df.describe()
# print("Description:", df_describe)


df_dublicated_count = df.duplicated().sum()
# print("Dublicates Count:",df_dublicated_count) // 0

value_counts_HeartDisease = df['HeartDisease'].value_counts().plot(kind='bar')
# print(value_counts_HeartDisease)
# plt.show()


def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)


plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)
plt.tight_layout()  #No Overlapping
# plt.show()


# Resting Bp can't be zero also Cholestrole as per google search i.e wrong data

# Replacing 0s in cholesterol featuer with its mean
cholesterol_mean = df.loc[
    df['Cholesterol'] != 0,
    'Cholesterol'
].mean()

# print(cholesterol_mean)

df['Cholesterol'] = df['Cholesterol'].replace(
    0,
    cholesterol_mean
)

df['Cholesterol'] = df['Cholesterol'].round(2)
# print(df['Cholesterol'])


# Same For restingBP
restingBP_mean = df.loc[
    df['RestingBP'] != 0,
    'RestingBP'
].mean()

# print(restingBP_mean)

df['RestingBP'] = df['RestingBP'].replace(
    0,
    restingBP_mean
)

df['RestingBP'] = df['RestingBP'].round(2)
# print(df['RestingBP'])


plotting('Age', 1)
plotting('RestingBP', 2)
plotting('Cholesterol', 3)
plotting('MaxHR', 4)
plt.tight_layout()
# plt.show()


sns.countplot(x=df['Sex'], hue=df['HeartDisease'])
# plt.show()

sns.countplot(x=df['ChestPainType'], hue=df['HeartDisease'])
# plt.show()

sns.countplot(x=df['FastingBS'], hue=df['HeartDisease'])
# plt.show()

sns.boxplot(x='HeartDisease', y='Cholesterol', data=df)
# plt.show()

sns.violinplot(x='HeartDisease', y='Age', data=df)
# plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()


#=======================
#Data Preprocessing

df_encode = pd.get_dummies(df, drop_first=True)
df_encode = df_encode.astype(int)
# print(df_encode)


from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier


X = df_encode.drop('HeartDisease', axis=1)
y = df_encode['HeartDisease']


# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)


# Scaling

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC()
}


result = []


for name, model in models.items():

    model.fit(X_train_scaled, y_train)

    y_predictions = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_predictions)

    f1 = f1_score(y_test, y_predictions)

    result.append({
        'model': name,
        'Accuracy': round(acc, 4),
        'F1 score': round(f1, 4)
    })


# print(result)
# [
#  {'model': 'Logistic Regression', 'Accuracy': 0.8713, 'F1 score': 0.887}, 
#  {'model': 'KNN', 'Accuracy': 0.8449, 'F1 score': 0.863}, 
#  {'model': 'Naive Byeas', 'Accuracy': 0.8581, 'F1 score': 0.8746}, 
#  {'model': 'Decision Tree', 'Accuracy': 0.736, 'F1 score': 0.759}, 
#  {'model': 'SVM', 'Accuracy': 0.8614, 'F1 score': 0.88}
# ]


# Classification Report

# for name, model in models.items():

#     y_predictions = model.predict(X_test_scaled)

#     print("\n==============================")
#     print(name)
#     print("==============================")

#     print(
#         classification_report(
#             y_test,
#             y_predictions
#         )
#     )


# Save Model

import joblib

joblib.dump(
    models['Logistic Regression'],
    'logistic_regression_model.pkl'
)

joblib.dump(
    scaler,
    'scaler.pkl'
)

joblib.dump(
    X.columns.to_list(),
    'columns.pkl'
)





