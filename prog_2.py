import pandas as pd
import Orange

def load_dataset(data):
    if isinstance(data, str):
        # Load from CSV file
        df = pd.read_csv(data)
    else:
        df = data

    attributes = []
    for col in df.columns[:-1]:  
        if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
            # Use the unique values as categories
            values = list(map(str, df[col].unique()))
            attributes.append(Orange.data.DiscreteVariable(col, values))
        else:
            attributes.append(Orange.data.ContinuousVariable(col))

    # Target variable (last column)
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
    # Path to the CSV file
    csv_path = "playtennis.csv"
    
    # Load and display the CSV data
    print("Loading PlayTennis dataset from CSV file...")
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print(f"\nTarget classes: {df['PlayTennis'].unique()}")
    
    # Convert to Orange table
    table = load_dataset(csv_path)
    print("\nOrange table created successfully!")

    print("\n" + "="*50)
    print("=== CN2 RULES ===")
    print("="*50)
    cn2_classifier = apply_cn2_learner(table)
    display_rules(cn2_classifier)

    print("\n" + "="*50)
    print("=== FOIL-LIKE RULES ===")
    print("="*50)
    foil_classifier = apply_foil_like_learner(table)
    display_rules(foil_classifier)

if __name__ == "__main__":
    main()
