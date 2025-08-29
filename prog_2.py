import pandas as pd
import Orange
from sklearn.datasets import load_iris


def load_iris_dataframe():
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df


def load_dataset(data):
    if isinstance(data, str):
        df = pd.read_csv(data)
    else:
        df = data

    attributes = []
    for col in df.columns[:-1]:  
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
           
            values = list(map(str, df[col].unique()))
            attributes.append(Orange.data.DiscreteVariable(col, values))
        else:
            attributes.append(Orange.data.ContinuousVariable(col))

  
    class_col = df.columns[-1]
    class_values = list(map(str, df[class_col].unique()))
    class_var = Orange.data.DiscreteVariable(class_col, class_values)

    domain = Orange.data.Domain(attributes, class_var)
    data_as_str = df.astype(str).values.tolist()
    table = Orange.data.Table.from_list(domain, data_as_str)

    return table

def apply_cn2_learner(table):
    learner = Orange.classification.rules.CN2Learner()
    classifier = learner(table)
    return classifier

def apply_foil_like_learner(table):
    learner = Orange.classification.rules.CN2SDUnorderedLearner()
    classifier = learner(table)
    return classifier

def display_rules(classifier):
    print("\nLearned Rules:\n")
    for rule in classifier.rule_list:
        print(rule)

def main():
   
    iris_df = load_iris_dataframe()
    table = load_dataset(iris_df)

    print("=== CN2 RULES ===")
    cn2_classifier = apply_cn2_learner(table)
    display_rules(cn2_classifier)

    print("\n=== FOIL-LIKE RULES ===")
    foil_classifier = apply_foil_like_learner(table)
    display_rules(foil_classifier)

if __name__ == "__main__":
    main()
