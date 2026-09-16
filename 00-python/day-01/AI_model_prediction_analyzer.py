
def load_data():
    predictions = [88, 91, 72, 63, 55, 97, 81, 76]
    print("Loading data from source")
    return predictions

def calculate_average(data):
    print("Calculating average of prediction")
    return sum(data)/len(data)

def highest_data(data):
    return max(data)

def lowest_data(data):
    return min(data)

def classify_prediction(score):
    if score >= 90:
        return "Excellent"
    elif score >=75:
        return "Good"
    elif score >=60:
        return "Average"
    else:
        return "Poor"

def classify_predictions(data):
    results = []
    for score in data:
        category = classify_prediction(score)
        results.append(category)
    return results

def category_count(results):
    counts = {
        "Excellent": 0,
        "Good" : 0,
        "Average" : 0,
        "Poor" : 0
    }    

    for result in results :
        counts[result] += 1

    return counts

def confidence_status(average):
    if average >= 90:
        return "Very Strong"
    elif average >=75:
        return "Strong"
    elif average >=60:
        return "Moderate"
    else:
        return "Weak"

def count_high_confidence(data, average):
    count = 0
    for score in data:
        if score >= average:
            count += 1
    return count    

def generate_report(data,average,highest,lowest,counts):
    print("\n === AI Prediction Report === ")

    print("Predictions:",data)

    print("Average Confidence:", average)

    print("Highest Confidence:", highest)

    print("Lowest Confidence:", lowest)

    print("Overall Confidence Status:", confidence_status(average))

    print("\nCategory Summary:")

    for category,count in counts.items():
        print(category,":",count)

    print("\nHigh Confidence Predictions Count:", count_high_confidence(data, average))

def main():
    data = load_data()

    average = calculate_average(data)

    highest = highest_data(data)

    lowest = lowest_data(data)

    results = classify_predictions(data)

    counts = category_count(results)

    generate_report(
        data,
        average,
        highest,
        lowest,
        counts
    )

main()

