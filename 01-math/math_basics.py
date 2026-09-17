import numpy as np

features = np.array([
    [5, 85],
    [7, 92],
    [3, 75],
    [8, 95]
])

weights = np.array([5, 0.4])

bias = 2

# Calculate the prediction for each student.
predictions = features @ weights + bias

print("Features:\n", features)

print("Weights:", weights)

print("Predictions:", predictions)

X = np.array([100, 5])
W = np.array([[5], [3]])

prediction = X @ W + bias

print("Prediction:", prediction) # 517