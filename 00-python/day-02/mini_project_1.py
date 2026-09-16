import numpy as np

predictions = np.array([45, 78, 92, 34, 88, 67, 95, 56, 73, 81])

print("predictions : ",predictions)

print("Average : ",np.mean(predictions))

print("Highest : ", np.max(predictions))

print("lowest : ",np.min(predictions))

high_confidence = predictions[predictions>=80]

print("High Confidence : ",high_confidence)

print("High Confidence Count : ",np.sum(predictions >=80))

low_confidence = predictions[predictions < 60]

print("Low Confidence : ",low_confidence)

print("Low Confidence Count : ",np.sum(predictions < 60))

print("dataSet Shape : ", np.shape(predictions))

print("Standard Deviation : ", np.std(predictions))