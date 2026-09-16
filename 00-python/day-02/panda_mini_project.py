import pandas as pd

data = {
    "Name": ["Amit", "Rahul", "Priya", "Sneha", "Vikas", "Neha"],
    "Age": [20, 21, 20, 22, 23, 21],
    "StudyHours": [5, 7, 3, 8, 2, 6],
    "Score": [82, 91, 67, 95, 54, 88]
}

df = pd.DataFrame(data)

print(df)

print("Shape:", df.shape)  # Display the shape of the DataFrame (rows, columns)

print("Column Names:", df.columns.to_list())

print("Data Types:", df.dtypes)

print("Only Names and Score:", df[["Name", "Score"]])

# Calculate highest score, avg score and lowest score
print("Highest Score:", df["Score"].max())
print("Average Score:", df["Score"].mean()) 
print("Lowest Score:", df["Score"].min())   

# Find Students with Score greater than 80
high_score_students = df[df["Score"] > 80]
print("Students with Score greater than 80:")
print(high_score_students)  

# Find Students with StudyHours >= 6
high_study_students = df[df["StudyHours"] >= 6]
print("Students with StudyHours >= 6:")
print(high_study_students)  

# Find Students with Score greater than 80 and StudyHours >= 6
high_score_high_study_students = df[(df["Score"] > 80) & (df["StudyHours"] >= 6)]
print("Students with Score greater than 80 and StudyHours >= 6:")   
print(high_score_high_study_students)

# Sort students by Score from highest to lowest.
sorted_students = df.sort_values(by="Score", ascending=False)
print("Students sorted by Score (highest to lowest):")
print(sorted_students)

# Create passed column based on Score >= 60
df["Passed"] = df["Score"] >= 60
print("DataFrame with Passed column:") 
print(df)

# Create a Category column
def classify_prediction(score):
    if score >= 90:
        return "Excellent"
    elif score >=75:
        return "Good"
    elif score >=60:
        return "Average"
    else:
        return "Poor"

df["Category"] = df["Score"].apply(classify_prediction)
print("DataFrame with Category column:")    
print(df)

print(df.describe() ) # Display summary statistics of the DataFrame