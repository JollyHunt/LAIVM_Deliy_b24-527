import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size) * np.sqrt(2./input_size)
        self.bias = 0.0
        self.loss_history = []
        self.acc_history = []

    def sigmoid(self, z):
        mask = z >= 0
        pos = np.zeros_like(z)
        neg = np.zeros_like(z)

        pos[mask] = 1. / (1. + np.exp(-z[mask]))
        neg[~mask] = np.exp(z[~mask]) / (1. + np.exp(z[~mask]))

        return pos + neg

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def compute_loss(self, y_pred, y_true):
        epsilon = 1e-12
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_accuracy(self, y_pred, y_true):
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions == y_true)

    def train(self, X, y, X_val, y_val, epochs=100, learning_rate=0.01, batch_size=256):
        m = X.shape[0]

        for epoch in range(epochs):
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                y_pred = self.forward(X_batch)

                error = y_pred - y_batch
                grad_w = np.dot(X_batch.T, error) / batch_size
                grad_b = np.mean(error)

                self.weights -= learning_rate * grad_w
                self.bias -= learning_rate * grad_b

            y_pred_train = self.forward(X)
            y_pred_val = self.forward(X_val)

            train_loss = self.compute_loss(y_pred_train, y)
            val_loss = self.compute_loss(y_pred_val, y_val)

            train_acc = self.compute_accuracy(y_pred_train, y)
            val_acc = self.compute_accuracy(y_pred_val, y_val)

            self.loss_history.append((train_loss, val_loss))
            self.acc_history.append((train_acc, val_acc))

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

def load_and_prepare_data():
    data = fetch_openml('credit-g', version=1, as_frame=True)
    df = data.frame

    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    y = (df['class_good'] == True).astype(int).values
    X = df.drop('class_good', axis=1).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def generate_large_dataset(X_original, y_original, n_samples=1000000):
    n_features = X_original.shape[1]

    cov_matrix = np.cov(X_original, rowvar=False)
    means = X_original.mean(axis=0)

    X_large = np.random.multivariate_normal(means, cov_matrix, size=n_samples)

    model = Perceptron(input_size=n_features)
    model.train(X_original, y_original, X_original, y_original, epochs=50)

    probas = model.forward(X_large)
    y_large = (probas > 0.5).astype(int)

    flip_idx = np.random.choice(n_samples, size=int(n_samples*0.05), replace=False)
    y_large[flip_idx] = 1 - y_large[flip_idx]

    return X_large, y_large


def plot_training_history(history):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot([h[0] for h in history.loss_history], label='Train Loss')
    plt.plot([h[1] for h in history.loss_history], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot([h[0] for h in history.acc_history], label='Train Acc')
    plt.plot([h[1] for h in history.acc_history], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

def visualize_data(X, y):
    X = X[:100000]
    y = y[:100000]
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.hist(X[y == 0, 0], bins=30, alpha=0.5, label='No Default')
    plt.hist(X[y == 1, 0], bins=30, alpha=0.5, label='Default')
    plt.title('Age Distribution')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.hist(X[y == 0, 1], bins=30, alpha=0.5, label='No Default')
    plt.hist(X[y == 1, 1], bins=30, alpha=0.5, label='Default')
    plt.title('Income Distribution')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.scatter(X[y == 0, 2], X[y == 0, 1], alpha=0.1, label='No Default')
    plt.scatter(X[y == 1, 2], X[y == 1, 1], alpha=0.1, label='Default')
    plt.title('Credit History vs Income')
    plt.xlabel('Credit History')
    plt.ylabel('Income')
    plt.legend()

    plt.subplot(2, 2, 4)
    class_counts = pd.Series(y).value_counts()
    class_counts.plot(kind='bar')
    plt.title('Class Distribution')
    plt.xticks([0, 1], ['No Default', 'Default'], rotation=0)

    plt.tight_layout()
    plt.show()

X, y = load_and_prepare_data()

X_large, y_large = generate_large_dataset(X, y, n_samples=1000000)

visualize_data(X_large, y_large)

X_train, X_test, y_train, y_test = train_test_split(
    X_large, y_large, test_size=0.2, random_state=42)

input_size = X_train.shape[1]
perceptron = Perceptron(input_size)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

perceptron.train(X_train, y_train, X_val, y_val,
                epochs=100, learning_rate=0.01, batch_size=512)

plot_training_history(perceptron)

y_pred_test = (perceptron.forward(X_test) > 0.5).astype(int)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred_test))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_test))
print("\nClassification Report:\n", classification_report(y_test, y_pred_test))