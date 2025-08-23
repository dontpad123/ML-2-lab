import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# RIPPER-like model (uses entropy)
ripper_model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
ripper_model.fit(X_train, y_train)

# Show rules
print("RIPPER-like Rules:")
print(tree.export_text(ripper_model, feature_names=feature_names))

# FOIL-like model (uses gini)
foil_model = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
foil_model.fit(X_train, y_train)

# Show rules
print("FOIL-like Rules:")
print(tree.export_text(foil_model, feature_names=feature_names))

# Make predictions
y_pred_ripper = ripper_model.predict(X_test)
y_pred_foil = foil_model.predict(X_test)

# Evaluate RIPPER
print("RIPPER Results:")
print(classification_report(y_test, y_pred_ripper, target_names=target_names))

# Evaluate FOIL
print("\nFOIL Results:")
print(classification_report(y_test, y_pred_foil, target_names=target_names))

# Simple accuracy comparison
ripper_acc = (y_pred_ripper == y_test).mean()
foil_acc = (y_pred_foil == y_test).mean()

print(f"RIPPER Accuracy: {ripper_acc:.2f}")
print(f"FOIL Accuracy: {foil_acc:.2f}")
