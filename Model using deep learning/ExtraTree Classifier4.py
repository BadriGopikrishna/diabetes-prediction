import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv("Enhanced_pima_diabetes.csv")

# Encode categorical features
label_encoders = {}
categorical_cols = ['BMI_Category', 'Age_Group']
existing_cols = [col for col in categorical_cols if col in df.columns]

for col in existing_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Feature Scaling
scaler = StandardScaler()
X[X.columns] = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Check class distribution before applying SMOTE
class_counts = y_train.value_counts()
minority_class = class_counts.idxmin()
majority_class = class_counts.idxmax()
minority_count = class_counts.min()
majority_count = class_counts.max()

# Apply SMOTE only if necessary
if minority_count / majority_count < 0.5:  # Avoid over-sampling if already balanced
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# Define ExtraTreesClassifier with optimized hyperparameters
extra_tree_clf = ExtraTreesClassifier(
    n_estimators=300,  # Increase trees for better learning
    max_depth=None,
    min_samples_split=4,  # Slightly decrease to find patterns
    min_samples_leaf=1,  # Allow smaller leaves for flexibility
    bootstrap=True,  # Bootstrap sampling for robustness
    random_state=42
)

# Train the model
extra_tree_clf.fit(X_train, y_train)

# Make predictions
y_train_pred = extra_tree_clf.predict(X_train)
y_test_pred = extra_tree_clf.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"Extra Trees Classifier Training Accuracy: {train_accuracy:.4f}")
print(f"Extra Trees Classifier Test Accuracy: {test_accuracy:.4f}")
print("Classification Report (Test Data):")
print(classification_report(y_test, y_test_pred))
