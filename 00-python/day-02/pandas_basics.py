import pandas as pd

# print(pd.__version__)

# SERIES
scores = pd.Series([82, 91, 67, 95, 88, 54, 97, 73])
print("Scores Series:")
print(scores)

# DATAFRAME
data = {
    "Name": ["Arun", "Rahul", "Priya", "Sneha"],
    "Age": [20, 21, 20, 22],
    "StudyHours": [5, 7, 3, 8],
    "Score": [82, 91, 67, 95]
}

df = pd.DataFrame(data)
print("\nDataFrame:")
print(df)

print(df.head(2)) # Display the first 2 rows of the DataFrame
# print(df.head()) # Display the first 5 rows of the DataFrame

# print(df.tail()) # Display the last 5 rows of the DataFrame
print(df.tail(2)) # Display the last 2 rows of the DataFrame

print(df.info())
print(df.notnull())