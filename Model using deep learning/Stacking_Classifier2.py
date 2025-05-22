import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Enhanced_pima_diabetes.csv")

# Define features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models with hyperparameter tuning
base_models = [
    ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('log_reg', LogisticRegression(C=1.0, max_iter=500))
]

# Define the stacking classifier with GradientBoosting as the final estimator
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42)
)

# Train the stacking model
stacking_clf.fit(X_train, y_train)

# Evaluate on the train data
y_train_pred = stacking_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Train Accuracy: {train_accuracy:.4f}")

# Make predictions on the test set
y_test_pred = stacking_clf.predict(X_test)

# Evaluate the model on the test data
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_test_pred))
