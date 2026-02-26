from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import logreg
import numpy as np

# Create a toy dataset: 1000 samples, 10 features
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# IMPORTANT: Logistic Regression is sensitive to feature scaling!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize the model with specific tuning
model = LogisticRegression(
    penalty='elasticnet',          # Using L2 (Ridge) regularization
    C=0.5,                 # Moderate regularization
    l1_ratio=0.5,
    solver='saga',         # Fast for large datasets and versatile
    max_iter=2000,         # Give it enough time to find the minimum
    multi_class='auto',    # Handles binary or multiclass automatically
    class_weight='balanced' # Useful if one class has way fewer samples than the other
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

print("--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

lr = 0.01
W,b = logreg.train(X_train, y_train, lr, 2000)
y_pred = logreg.predict(X_test, W, b)

print("\n--- Accuracy using logreg from scratch ---")
print(np.mean(y_pred == y_test))

