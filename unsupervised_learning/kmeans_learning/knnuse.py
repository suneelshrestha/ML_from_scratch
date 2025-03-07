from knnlearn import KNN
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn import metrics

# Load dataset
datasets = load_breast_cancer()

df = pd.DataFrame(datasets.data, columns=datasets.feature_names)

# Standardize the data
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df)
scaled_df = pd.DataFrame(scaled_df, columns=datasets.feature_names)

X = scaled_df.values  # Convert DataFrame to NumPy array
y = datasets.target

# Split into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=23, test_size=0.2)

# Create KNN model
knn = KNN(k = 3)  # Set k=3
knn.fit(X_train, Y_train)

# Predict on test data
predictions = knn.predict(X_test)

# Calculate accuracy
accuracy = metrics.accuracy_score(Y_test, predictions)
print("Accuracy:", accuracy)

# Display some predictions
for i in range(10):
    print(f"Actual value: {Y_test[i]}, Prediction: {predictions[i]}")
