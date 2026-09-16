import numpy as np

#Age, study hours, score
students = np.array([
    [20, 5, 82],
    [21, 7, 91],
    [20, 3, 67],
    [22, 8, 95],
    [21, 6, 88],
    [23, 2, 54],
    [22, 9, 97],
    [20, 4, 73]
])

#Student Shape
print("Students Shape:", np.shape(students))

#Average Age
print("Average Age:", np.mean(students[:,0]))

#Average Study Hours
print("Average Study Hours:", np.mean(students[:,1]))

#Average Score
print("Average Score:", np.mean(students[:,2]))

#Highest Score
print("Highest Score:", np.max(students[:,2])) 

#Lowest Score
print("Lowest Score:", np.min(students[:,2]))

#Students With Score >= 80
print("Students With Score >= 80:", students[students[:,2] >= 80])

# Students with Study Hours >= 6
print("Students with Study Hours >= 6:", students[students[:,1] >= 6 ])

# Number of High Scoring Students
print("Number of High Scoring Students:", np.sum(students[:,2] >=80))

# Number of Low Scoring Students
print("Number of Low Scoring Students:", np.sum(students[:,2] < 60))

# Average score of students who studied >= 6 hours
# high_study_time = students[students[:,1] >= 6]
# print("Average score of students who studied >= 6 hours:", np.mean(high_study_time[:,2]))
print("Average score of students who studied >= 6 hours:", np.mean(students[students[:,1] >= 6][:,2]))