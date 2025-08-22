import pandas as pd
from typing import List, Tuple

def get_input_data():
    print("Enter the dataset row-by-row as comma-separated values (e.g., Sunny,Warm,Normal,Strong,Warm,Same,Yes).")
    print("Enter an empty line to finish.\n")
    dataset = []
    while True:
        row = input("Enter example: ").strip()
        if not row:
            break
        parts = row.split(",")
        dataset.append(parts)
    return dataset


def get_predefined_data():
    return [
        ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
        ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
        ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
    ]

def find_s_algorithm(data):
    print("\n=== Find-S Algorithm ===")
    hypothesis = None
    for row in data:
        *attributes, label = row
        if label.lower() == "yes":
            if hypothesis is None:
                hypothesis = attributes.copy()
            else:
                for i in range(len(attributes)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = '?'
    print("Final Hypothesis (Find-S):", hypothesis)
    return hypothesis


def more_general(h1, h2):
    """Check if h1 is more general than h2"""
    for x, y in zip(h1, h2):
        if x != '?' and x != y:
            return False
    return True

def generalize_S(example, S):
    """Generalize the specific boundary"""
    for i in range(len(S)):
        if S[i] != example[i]:
            S[i] = '?'
    return S

def specialize_G(example, G, attributes):
    """Specialize the general boundary - CORRECTED VERSION"""
    new_G = []
    for g in G:
        
        if all(g[i] == '?' or g[i] == example[i] for i in range(len(example))):
            
            for i in range(len(g)):
                if g[i] == '?':
                    for value in attributes[i]:
                        if value != example[i]:
                            new_h = g.copy()
                            new_h[i] = value
                            
                            if not any(more_general(new_h, existing_h) for existing_h in new_G):
                                new_G.append(new_h)
        else:
     
            new_G.append(g)
    

    final_G = []
    for h in new_G:
        if not any(more_general(other_h, h) for other_h in new_G if other_h != h):
            final_G.append(h)
    
    return final_G


def candidate_elimination_algorithm(data):
    print("\n=== Candidate Elimination Algorithm ===")
    

    attributes = []
    for i in range(len(data[0]) - 1):
        unique_vals = list(set(row[i] for row in data))
        attributes.append(unique_vals)
    
    
    S = ['0'] * (len(data[0]) - 1)  
    G = [['?'] * (len(data[0]) - 1)] 
    
    print("Initial S:", S)
    print("Initial G:", G)
    print()
    
    for idx, row in enumerate(data):
        *x, label = row
        print(f"Processing example {idx+1}: {x} -> {label}")
        
        if label.lower() == 'yes':  
            
            G = [g for g in G if all(g[i] == '?' or g[i] == x[i] for i in range(len(x)))]
            
           
            if S == ['0'] * len(x):
                S = x.copy()
            else:
                S = generalize_S(x, S)
            
            
            G = [g for g in G if more_general(g, S)]
            
        else:  
            if all(S[i] == x[i] or S[i] == '?' for i in range(len(x))):
                S = ['0'] * len(x)
            
            
            G = specialize_G(x, G, attributes)
            
          
            G = [g for g in G if more_general(g, S)]
        
        print(f"After example {idx+1}:")
        print(f"S: {S}")
        print(f"G: {G}")
        print()
    
    print("Final Specific Hypothesis (S):", S)
    print("Final General Hypotheses (G):", G)
    return S, G






data = get_predefined_data()

print("Dataset:")
for i, row in enumerate(data):
    print(f"Example {i+1}: {row}")


final_hypothesis = find_s_algorithm(data)


S_final, G_final = candidate_elimination_algorithm(data)

print("\n=== Final Results ===")
print("Find-S Final Hypothesis:", final_hypothesis)
print("Candidate Elimination Version Space:")
print("Most Specific Hypothesis (S):", S_final)
print("Most General Hypotheses (G):")
for i, g in enumerate(G_final):
    print(f"  G{i+1}: {g}")
