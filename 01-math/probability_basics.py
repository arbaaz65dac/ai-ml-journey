import numpy as np

predictions = np.array([
    92, 85, 76, 95, 61,
    88, 73, 90, 55, 82,
    79, 96, 68, 87, 91
])

def analyze_predictions(predictions):
    total_predictions = len(predictions)
    average_prediction = np.mean(predictions)
    variance = np.var(predictions)
    std_deviation = np.std(predictions)

    high_confidence_count = np.sum(predictions >= 80)
    low_confidence_count = np.sum(predictions < 60)

    probability_high = high_confidence_count / total_predictions
    probability_low = low_confidence_count / total_predictions

    highest_prediction = np.max(predictions)
    lowest_prediction = np.min(predictions)

    return {
        "Total Predictions": total_predictions,
        "Average Prediction": average_prediction,
        "Variance": variance,
        "Standard Deviation": std_deviation,
        "High Confidence (>=80)": high_confidence_count,
        "Low Confidence (<60)": low_confidence_count,
        "Probability High": probability_high,
        "Probability Low": probability_low,
        "Highest Prediction": highest_prediction,
        "Lowest Prediction": lowest_prediction
    }

if __name__ == "__main__":
    results = analyze_predictions(predictions)
    for key, value in results.items():
        print(f"{key}: {value}")