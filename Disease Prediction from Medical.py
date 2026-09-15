import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
cancer = load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = cancer.target

# Keep top 10 most relevant features for a clean user interface
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area', 
    'mean smoothness', 'mean compactness', 'mean concavity', 
    'mean concave points', 'mean symmetry', 'mean fractal dimension'
]
X_subset = X[selected_features]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42, stratify=y)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Save model, scaler, and feature names
joblib.dump(model, 'disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(selected_features, 'feature_names.pkl')

print("Model trained and saved successfully!")