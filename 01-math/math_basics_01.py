import numpy as np

A = np.array([10, 20, 30])
B = np.array([2, 4, 6])

#Addition
print("Addition:\n", A + B)

#Subtraction
print("Subtraction:\n", A - B)

#Multiplication
print("Multiplication:\n", A * B)

#Dot product
print("Dot Product:\n", A @ B)


# age study_hours score
students = np.array([
    [20, 5, 80],
    [21, 7, 90],
    [22, 3, 70],
    [20, 8, 95],
    [23, 6, 85]
])

# Shape
print("Shape:", students.shape)

# Average age
print("Average Age:", np.average(students[:, 0]))

# Average study hours
print("Average Study Hours:", np.average(students[:, 1]))

# Average score
print("Average Score:", np.average(students[:, 2]))

# Highest score
print("Highest Score:", np.max(students[:, 2]))

# Lowest score
print("Lowest Score:", np.min(students[:, 2]))

# Students with score ≥ 80
print("Students with Score ≥ 80:")
print(students[students[:, 2] >= 80])

# Average score using axis=0
print("Average Score (axis=0):", np.average(students, axis=0)[2])

# Sum of each student's features using axis=1
print("Sum of Each Student's Features (axis=1):", np.sum(students, axis=1))