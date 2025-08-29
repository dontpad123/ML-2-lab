!pip install orange3 scikit-learn pandas

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import Orange
from Orange.classification.rules import CN2Learner, CN2SDUnorderedLearner

iris = load_iris()
X = iris.data
y = iris.target


df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("Dataset loaded successfully!")
print("Shape:", df.shape)
df.head()


def load_dataset(df):
    attributes = []
    for col in df.columns[:-1]:  # All columns except the target
        if df[col].dtype == 'object':
            # Use the unique values as categories
            values = list(map(str, df[col].unique()))
            attributes.append(Orange.data.DiscreteVariable(col, values))
        else:
            attributes.append(Orange.data.ContinuousVariable(col))

   
    class_col = df.columns[-1]
    class_values = list(map(str, df[class_col].unique()))
    class_var = Orange.data.DiscreteVariable(class_col, class_values)

    domain = Orange.data.Domain(attributes, class_var)

    # Convert all values to strings (because Orange expects matching categories)
    data_as_str = df.astype(str).values.tolist()
    table = Orange.data.Table.from_list(domain, data_as_str)

    return table


table = load_dataset(df)
print("Orange table created successfully!")

def apply_cn2_learner(table):
    learner = CN2Learner()
    classifier = learner(table)
    return classifier
def apply_foil_like_learner(table):
    """
    Apply FOIL-like algorithm (CN2SDUnorderedLearner) for rule learning
    """
    learner = CN2SDUnorderedLearner()
    classifier = learner(table)
    return classifier

def display_rules(classifier):
    print("\nLearned Rules:\n")
    for rule in classifier.rule_list:
        print(rule)

def main():
    print("=== CN2 RULES ===")
    cn2_classifier = apply_cn2_learner(table)
    display_rules(cn2_classifier)

    print("\n=== FOIL-LIKE RULES ===")
    foil_classifier = apply_foil_like_learner(table)
    display_rules(foil_classifier)

if __name__ == "__main__":
    main()

main()
