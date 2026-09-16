import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    data = {
    "Name": [
        "Amit", "Rahul", "Priya", "Sneha", "Vikas",
        "Neha", "Rohan", "Anjali", "Karan", "Pooja",
        "Aditya", "Simran"
    ],

    "Age": [
        20, 21, 20, 22, 23,
        21, 22, 20, 23, 21,
        22, 20
    ],

    "StudyHours": [
        5, 7, 3, 8, 2,
        6, 4, 9, 3, 5,
        7, 2
    ],

    "Attendance": [
        85, 92, 75, 95, 60,
        88, 72, 98, 68, 80,
        90, 65
    ],

    "Score": [
        82, 91, 67, 95, 54,
        88, 73, 97, 61, 78,
        89, 50
    ]
}
    # df = pd.DataFrame(data)
    return pd.DataFrame(data) 

def inspect_data(df):
    print("Number of Rows and Columns:",df.shape)
    print("Column Names:",df.columns.tolist())
    print("Data Types:",df.dtypes)
    print("Missing Values:\n",df.isnull().sum())
    print("Basic Statics:\n",df.describe())

# print(inspect_data(load_data()))


def analyze_data(df):

    average_score = df["Score"].mean()
    highest_score = df["Score"].max()
    lowest_score = df["Score"].min()

    average_study_hours = df["StudyHours"].mean()
    average_attendance = df["Attendance"].mean()

    top_student = df.loc[df["Score"].idxmax()]
    lowest_student = df.loc[df["Score"].idxmin()]

    return (
        average_score,
        highest_score,
        lowest_score,
        average_study_hours,
        average_attendance,
        top_student,
        lowest_student
    )

def classify_score(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Average"
    else:
        return "Poor"

def classify_students(df):

    df["Passed"] = df["Score"] >= 60

    df["Category"] = df["Score"].apply(classify_score)

    return df

def generate_report(df, results):

    (
        average_score,
        highest_score,
        lowest_score,
        average_study_hours,
        average_attendance,
        top_student,
        lowest_student
    ) = results

    # Passed and failed students
    passed_students = df["Passed"].sum()
    failed_students = (~df["Passed"]).sum()

    # Students studying >= 6 hours
    high_study_students = df[df["StudyHours"] >= 6]
    high_study_average = high_study_students["Score"].mean()

    # Students with attendance >= 85%
    high_attendance_students = df[df["Attendance"] >= 85]
    high_attendance_average = high_attendance_students["Score"].mean()

    # Performance category counts
    category_counts = df["Category"].value_counts()

    print("       STUDENT PERFORMANCE REPORT")

    print("\nTotal Students       :", len(df))
    print("Average Score        :", round(average_score, 2))
    print("Highest Score        :", highest_score)
    print("Lowest Score         :", lowest_score)

    print("\nAverage Study Hours  :", round(average_study_hours, 2))
    print("Average Attendance   :", round(average_attendance, 2))

    print("\nPassed Students      :", passed_students)
    print("Failed Students      :", failed_students)

    print("\nTop Student          :", top_student["Name"])
    print("Top Student Score    :", top_student["Score"])

    print("\nLowest Performer     :", lowest_student["Name"])
    print("Lowest Student Score :", lowest_student["Score"])

    print("\nStudents Studying >= 6 Hours :", len(high_study_students))
    print("Average Score                :", round(high_study_average, 2))

    print("\nStudents Attendance >= 85%   :", len(high_attendance_students))
    print("Average Score                :", round(high_attendance_average, 2))

    print("\n------------- CATEGORIES -------------")

    print("Excellent :", category_counts.get("Excellent", 0))
    print("Good      :", category_counts.get("Good", 0))
    print("Average   :", category_counts.get("Average", 0))
    print("Poor      :", category_counts.get("Poor", 0))

    print("\n============================================")
def visualize_data(df):

    
    # Visualization 1
    # Study Hours vs Score

    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["StudyHours"],
        df["Score"]
    )

    plt.xlabel("Study Hours")
    plt.ylabel("Score")
    plt.title("Study Hours vs Score")

    plt.grid()

    plt.show()


    # Visualization 2
    # Student vs Score
   
    plt.figure(figsize=(10, 5))

    plt.bar(
        df["Name"],
        df["Score"]
    )

    plt.xlabel("Student")
    plt.ylabel("Score")
    plt.title("Student Scores")

    plt.xticks(rotation=45)

    plt.grid(axis="y")

    plt.show()


    # Visualization 3
    # Score Distribution

    plt.figure(figsize=(8, 5))

    plt.hist(df["Score"])

    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.title("Score Distribution")

    plt.grid()

    plt.show()


    # Visualization 4
    # Performance Categories

    category_counts = df["Category"].value_counts()

    plt.figure(figsize=(8, 5))

    plt.bar(
        category_counts.index,
        category_counts.values
    )

    plt.xlabel("Performance Category")
    plt.ylabel("Number of Students")
    plt.title("Performance Category Distribution")

    plt.grid(axis="y")

    plt.show()



def main():

    df = load_data()

    inspect_data(df)

    results = analyze_data(df)

    df = classify_students(df)

    generate_report(df, results)

    visualize_data(df)

main()