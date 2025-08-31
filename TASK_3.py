# Task 03: Decision Tree Classifier - Bank Marketing Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# -----------------------
# Load Dataset
# -----------------------
# Make sure you have "bank.csv" in your working directory (download from UCI repo)
df = pd.read_csv("bank.csv", sep=";")  # UCI dataset uses semicolon delimiter

print("Dataset shape:", df.shape)
print("Dataset preview:\n", df.head())

# -----------------------
# Data Preprocessing
# -----------------------
# Encode categorical variables into numeric
df_encoded = pd.get_dummies(df, drop_first=True)

# Features (X) and Target (y)
X = df_encoded.drop("y_yes", axis=1)   # 'y' column converted to y_yes after encoding
y = df_encoded["y_yes"]

# Split into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------
# Decision Tree Classifier
# -----------------------
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# -----------------------
# Evaluation
# -----------------------
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------
# Visualization
# -----------------------
plt.figure(figsize=(20,10))
plot_tree(
    clf,
    filled=True,
    feature_names=X.columns,
    class_names=["No", "Yes"]
)
plt.show()
