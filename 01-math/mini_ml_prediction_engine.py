import numpy as np

X = np.array([
    [5, 85],
    [7, 92],
    [3, 75],
    [8, 95],
    [6, 88]
])

W = np.array([4, 0.5])

bias = 10

print("Predictions:", X @ W + bias)

# Average prediction
print("Average Prediction:", np.average(X @ W + bias))

# Highest prediction
print("Highest Prediction:", np.max(X @ W + bias))

# Lowest prediction
print("Lowest Prediction:", np.min(X @ W + bias))

# Students with prediction >= 70
print("Students with Prediction >= 70:")
print(X[X @ W + bias >= 70])

# Number of predictions >= 70
print("Number of Predictions >= 70:", np.sum(X @ W + bias >= 70))


def classify_prediction(score):
    if score >= 90:
        return "Excellent"
    elif score >= 75:
        return "Good"
    elif score >= 60:
        return "Average"
    else:
        return "Poor"

# Apply the function to each prediction
predictions = X @ W + bias
categories = np.array([classify_prediction(score) for score in predictions])
print("Prediction Categories:", categories)

