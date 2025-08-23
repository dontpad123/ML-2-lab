import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names


data = pd.DataFrame(data=X, columns=feature_names)
data['species'] = pd.Series(y).map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset shape:", data.shape)
data.head()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")
print("Class distribution in training set:")
print(pd.Series(y_train).value_counts())

def generate_ripper_rules(X_train, y_train, feature_names, target_names):
    """
    Simulate RIPPER-like rule generation using decision tree with entropy
    and extracting rules from the tree structure
    """
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    
    rules = extract_rules_from_tree(clf, feature_names, target_names)
    
    return clf, rules

def extract_rules_from_tree(tree_model, feature_names, target_names):
    """Extract human-readable rules from decision tree"""
    n_nodes = tree_model.tree_.node_count
    children_left = tree_model.tree_.children_left
    children_right = tree_model.tree_.children_right
    feature = tree_model.tree_.feature
    threshold = tree_model.tree_.threshold
    
    rules = []
    
    def extract_rules_recursive(node_id, current_rule):
        if children_left[node_id] == children_right[node_id]:  # Leaf node
            class_id = np.argmax(tree_model.tree_.value[node_id])
            rules.append(f"{current_rule} → {target_names[class_id]}")
        else:
           
            left_rule = f"{current_rule} AND {feature_names[feature[node_id]]} <= {threshold[node_id]:.2f}"
            extract_rules_recursive(children_left[node_id], left_rule)
            
           
            right_rule = f"{current_rule} AND {feature_names[feature[node_id]]} > {threshold[node_id]:.2f}"
            extract_rules_recursive(children_right[node_id], right_rule)
    
    extract_rules_recursive(0, "IF")
    return rules


ripper_model, ripper_rules = generate_ripper_rules(X_train, y_train, feature_names, target_names)

def generate_foil_rules(X_train, y_train, feature_names, target_names):
    """
    Simulate FOIL-like rule generation using decision tree with gini impurity
    and extracting simplified rules
    """
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
   
    rules = extract_simplified_rules(clf, feature_names, target_names)
    
    return clf, rules

def extract_simplified_rules(tree_model, feature_names, target_names):
    """Extract simplified rules focusing on the most important features"""
    rules = []
    tree_rules_text = tree.export_text(tree_model, feature_names=feature_names)
    
    for line in tree_rules_text.split('\n'):
        if 'class:' in line:
            parts = line.split('class:')
            condition = parts[0].strip()
            class_id = int(parts[1].strip())
            rules.append(f"IF {condition} → {target_names[class_id]}")
    
    return rules


foil_model, foil_rules = generate_foil_rules(X_train, y_train, feature_names, target_names)

print("RIPPER-like Rules (simulated):")
print("=" * 50)
for i, rule in enumerate(ripper_rules, 1):
    print(f"Rule {i}: {rule}")

print("\n" + "=" * 50)
print("FOIL-like Rules (simulated):")
print("=" * 50)
for i, rule in enumerate(foil_rules, 1):
    print(f"Rule {i}: {rule}")


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
tree.plot_tree(ripper_model, feature_names=feature_names, 
               class_names=target_names, filled=True)
plt.title("RIPPER-like Decision Tree")

plt.subplot(1, 2, 2)
tree.plot_tree(foil_model, feature_names=feature_names, 
               class_names=target_names, filled=True)
plt.title("FOIL-like Decision Tree")

plt.tight_layout()
plt.show()


y_pred_ripper = ripper_model.predict(X_test)
y_pred_foil = foil_model.predict(X_test)

print("RIPPER-like Model Performance:")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred_ripper):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_ripper, target_names=target_names))

print("\n" + "=" * 40)
print("FOIL-like Model Performance:")
print("=" * 40)
print(f"Accuracy: {accuracy_score(y_test, y_pred_foil):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_foil, target_names=target_names))

# Compare both models
ripper_acc = accuracy_score(y_test, y_pred_ripper)
foil_acc = accuracy_score(y_test, y_pred_foil)

print("Model Comparison:")
print("=" * 30)
print(f"RIPPER-like Accuracy: {ripper_acc:.3f}")
print(f"FOIL-like Accuracy: {foil_acc:.3f}")

if ripper_acc > foil_acc:
    print("RIPPER-like model performed better")
elif foil_acc > ripper_acc:
    print("FOIL-like model performed better")
else:
    print("Both models performed equally")

