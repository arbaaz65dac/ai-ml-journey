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

df_info = df.info()
# print("Info:\n", df_info)

df_describe = df.describe()
# print("Description:", df_describe)


df_dublicated_count = df.duplicated().sum()
# print("Dublicates Count:",df_dublicated_count) // 0

value_counts_HeartDisease = df['HeartDisease'].value_counts().plot(kind='bar')
# print(value_counts_HeartDisease) 
# plt.show()

def plotting(var, num) :
    plt.subplot(2,2,num)
    sns.histplot(df[var], kde = True)

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)
plt.tight_layout()  #No Overlapping 
# plt.show()

# Resting Bp can't be zero also Cholestrole as per google search i.e wrong data

# Replacing 0s in cholesterol featuer with its mean 
cholesterol_mean = df.loc[df['Cholesterol'] !=0, 'Cholesterol'].mean() 
# print(cholesterol_mean)

df['Cholesterol'] = df['Cholesterol'].replace(0,cholesterol_mean)
df['Cholesterol'] = df['Cholesterol'].round(2)
# print(df['Cholesterol'])

# Same For restingBP
restingBP_mean = df.loc[df['RestingBP'] != 0, 'RestingBP'].mean()
# print(restingBP_mean)

df['RestingBP'] = df['RestingBP'].replace(0,restingBP_mean)
df['RestingBP'] = df['RestingBP'].round(2)
# print(df['RestingBP'])

plotting('Age',1)
plotting('RestingBP',2)
plotting('Cholesterol',3)
plotting('MaxHR',4)
plt.tight_layout()
# plt.show()

sns.countplot(x= df['Sex'], hue= df['HeartDisease'])
# plt.show()

sns.countplot(x= df['ChestPainType'], hue= df['HeartDisease'])
# plt.show()

sns.countplot(x= df['FastingBS'], hue= df['HeartDisease'])
# plt.show()

sns.boxplot(x='HeartDisease',y = 'Cholesterol', data= df)
# plt.show()

sns.violinplot(x='HeartDisease',y = 'Age', data= df)
# plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
# plt.show()

#=======================
#Data Preprocessing

df_encode = pd.get_dummies(df, drop_first= True)
df_encode = df_encode.astype(int)
# print(df_encode)
 
from sklearn.preprocessing import StandardScaler
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']

scaler = StandardScaler()
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
# print(df_encode.head())