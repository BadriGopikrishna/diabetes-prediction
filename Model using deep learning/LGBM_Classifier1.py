import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Enhanced_pima_diabetes.csv")

# Define features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LGBMClassifier with hyperparameter tuning
lgbm_clf = LGBMClassifier(
    n_estimators=200,  # Increase the number of estimators
    learning_rate=0.05,  # Decrease learning rate for better precision
    num_leaves=50,  # More leaves allows the model to be more complex
    max_depth=7,  # Restrict depth to prevent overfitting
    random_state=42
)

# Train the model
lgbm_clf.fit(X_train, y_train)

# Make predictions
y_train_pred = lgbm_clf.predict(X_train)
y_test_pred = lgbm_clf.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print classification report for test set
print("Classification Report (Test Data):")
print(classification_report(y_test, y_test_pred))
