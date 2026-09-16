import matplotlib.pyplot as plt

days = [1,2,3,4,5]
scores = [50,60,65,75,85]
study_hours = [2, 3, 4, 6, 8]

#  LINE CHART
plt.plot(days,scores)
plt.plot(days,study_hours)

plt.xlabel("days")
plt.ylabel("scores")
plt.title("Students Score progress")
plt.grid()
plt.show()

# BAR CHART
plt.bar(days,scores)
plt.xlabel("days")
plt.ylabel("scores")
plt.title("Students Score progress")
plt.show()

# SCATTER PLOT
plt.figure(figsize=(8, 5))
plt.scatter(days,scores)
plt.xlabel("days")
plt.ylabel("scores")
plt.title("Students Score progress")
plt.grid()
plt.show()

# HISTOGRAM
import matplotlib.pyplot as plt

scores1 = [45, 52, 55, 60, 62, 65, 67, 70,
          72, 75, 78, 80, 82, 85, 88, 90, 92, 95]

plt.hist(scores1)

plt.xlabel("Score")
plt.ylabel("Frequency")
plt.title("Score Distribution")

plt.show()

import matplotlib.pyplot as plt

epochs = [1, 2, 3, 4, 5]

training_accuracy = [60, 68, 75, 82, 88]
validation_accuracy = [58, 65, 72, 78, 84]

plt.plot(epochs, training_accuracy, label="Training")
plt.plot(epochs, validation_accuracy, label="Validation")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")

plt.legend()
plt.grid()

plt.show()