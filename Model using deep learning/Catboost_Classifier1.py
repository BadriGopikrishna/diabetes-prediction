import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Enhanced_pima_diabetes.csv")

# Encode categorical features
label_encoders = {}
for col in ['BMI', 'Age']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CatBoostClassifier with hyperparameter tuning
catboost_clf = CatBoostClassifier(
    iterations=500,          # Increase the number of iterations (trees)
    depth=10,                # Increased depth for more complex trees
    learning_rate=0.05,      # Lower learning rate for better precision
    l2_leaf_reg=10,          # L2 regularization to prevent overfitting
    border_count=128,        # Number of splits for continuous features
    verbose=100,             # Display progress every 100 iterations
    random_state=42
)

# Train the model
catboost_clf.fit(X_train, y_train)

# Make predictions
y_train_pred = catboost_clf.predict(X_train)
y_test_pred = catboost_clf.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print train and test accuracy
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print classification report for test set
print("Classification Report (Test Data):")
print(classification_report(y_test, y_test_pred))
