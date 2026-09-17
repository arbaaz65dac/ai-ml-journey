import numpy as np
import matplotlib.pyplot as plt

def predict(X, w, b):
    return X * w + b

def calculate_loss(y, predictions):
    return np.mean((y - predictions) ** 2)

def calculate_gradients(X, y, predictions):
    error = y - predictions
    gradient_w = (-2 / len(X)) * np.sum(X * error)
    gradient_b = (-2 / len(X)) * np.sum(error)
    return gradient_w, gradient_b

def train(X, y, learning_rate, iterations):
    w = 0.0
    b = 0.0
    loss_history = []
    for i in range(iterations):
        predictions = predict(X, w, b)
        loss = calculate_loss(y, predictions)
        gradient_w, gradient_b = calculate_gradients(X, y, predictions)
        w = w - learning_rate * gradient_w
        b = b - learning_rate * gradient_b
        loss_history.append(loss)
        if i % 100 == 0:
            print(f"Iteration: {i} | Loss: {loss:.4f} | Weight: {w:.4f} | Bias: {b:.4f}")
    return w, b, loss_history

def main():
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = np.array([42, 45, 51, 58, 62, 68, 73, 79, 85, 91], dtype=float)
    learning_rate = 0.01
    iterations = 1000

    print("========================================")
    print("       MINI ML TRAINING SYSTEM")
    print("========================================")

    print("\nInitial Weight: 0.0")
    print("Initial Bias: 0.0")

    w, b, loss_history = train(X, y, learning_rate, iterations)
    predictions = predict(X, w, b)
    final_loss = calculate_loss(y, predictions)

    print("\n========================================")
    print("          TRAINING COMPLETE")
    print("========================================")
    print(f"\nFinal Weight: {w:.4f}")
    print(f"Final Bias: {b:.4f}")
    print(f"Final Loss: {final_loss:.4f}")

    new_hours = np.array([11, 12, 13], dtype=float)
    new_predictions = predict(new_hours, w, b)

    print("\nPredictions for New Inputs:")
    for hours, prediction in zip(new_hours, new_predictions):
        print(f"{hours:.0f} hours → {prediction:.2f}")

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss")
    plt.show()

    plt.scatter(X, y)
    model_predictions = predict(X, w, b)
    plt.plot(X, model_predictions)
    plt.xlabel("Study Hours")
    plt.ylabel("Score")
    plt.title("Linear Regression")
    plt.show()

if __name__ == "__main__":
    main()
