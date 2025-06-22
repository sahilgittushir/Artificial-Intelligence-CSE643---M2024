import pickle
import pandas as pd
import bnlearn as bn
import os
import matplotlib.pyplot as plt
import numpy as np
from memory_profiler import memory_usage
import time
import networkx as nx
from test_model import test_model  


##########################
## Helper Functions ##
##########################

def load_data():
    """Load train and validation datasets from CSV files."""
    try:
        train_data = pd.read_csv("train_data.csv")
        val_data = pd.read_csv("validation_data.csv")
        print(f"[+] Data loaded successfully. Train Data Shape: {train_data.shape}, Validation Data Shape: {val_data.shape}")
    except FileNotFoundError as e:
        print(f"[!] File not found: {e}")
        return None, None
    return train_data, val_data

def make_fully_connected_dag(features):
    """
    Create a fully connected Directed Acyclic Graph (DAG) for the given features.
    """
    print("[+] Creating Fully Connected Base Model...")
    
    # Use topological order to avoid cycles
    edges = [(features[i], features[j]) for i in range(len(features)) for j in range(i + 1, len(features))]
    
    # Convert the list of edges into a DAG
    fully_connected_dag = bn.make_DAG(edges)
    
    return fully_connected_dag



def make_base_network(df):
    """
    Define and fit the initial Bayesian Network with all possible edges.
    """
    print("[+] Creating Fully Connected Base Model...")
    
    # Features of the dataset
    features = df.columns.tolist()
    
    # Create a fully connected DAG
    fully_connected_dag = make_fully_connected_dag(features)
    
    # Learn CPDs for the fully connected DAG
    print("[+] Learning CPDs for the Base Model...")
    base_model = bn.parameter_learning.fit(fully_connected_dag, df)
    
    print("[+] Base Model Created Successfully.")
    return base_model

def make_pruned_network(df):
    """
    Define and fit a pruned Bayesian Network.
    """
    print("[+] Pruning Bayesian Network...")
    
    # Create the initial DAG
    DAG = make_base_network(df)

    # Example pruning: Remove unnecessary edges
    print("[+] Pruning edges...")
    DAG['adjmat'].loc['Start_Stop_ID', 'End_Stop_ID'] = 0  # Example: Remove edge Start_Stop_ID -> End_Stop_ID

    # Convert the adjacency matrix to an edge list
    edge_list = []
    for source, targets in DAG['adjmat'].iterrows():
        for target, exists in targets.items():
            if exists:
                edge_list.append((source, target))

    print("[+] Pruned Edge List:")
    print(edge_list)

    # Create a new pruned DAG using the edge list
    pruned_DAG = bn.make_DAG(edge_list)

    # Perform parameter learning for the pruned DAG
    pruned_BN = bn.parameter_learning.fit(pruned_DAG, df)

    print("[+] Pruned Bayesian Network created successfully")
    return pruned_BN

def make_optimized_network(df):
    """
    Perform structure optimization and fit the optimized Bayesian Network.
    """
    print("[+] Optimizing Bayesian Network...")

    # Use Hill Climbing for optimization
    optimized_DAG = bn.structure_learning.fit(df, methodtype='hc')
    print(f"[+] Optimized DAG: {optimized_DAG}")

    # Perform parameter learning on the optimized DAG
    optimized_BN = bn.parameter_learning.fit(optimized_DAG, df)
    print(f"[+] Optimized Bayesian Network: {optimized_BN}")

    return optimized_BN

def plot_network(DAG, title="Bayesian Network"):
    """
    Plot the Bayesian Network DAG using networkx with better clarity for tasks.
    """
    # Convert the DAG adjacency matrix to a networkx graph
    G = nx.DiGraph(DAG['adjmat'].values)

    # Set node labels from DAG variables
    node_labels = {i: node for i, node in enumerate(DAG['adjmat'].index)}
    nx.relabel_nodes(G, node_labels, copy=False)
    
    # Plot the network with customized appearance
    pos = nx.spring_layout(G, seed=42)  # Use spring layout for clear separation
    plt.figure(figsize=(12, 10))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=12, font_weight='bold', arrows=True)
    plt.title(title, fontsize=16)
    plt.show()

def save_model(fname, model):
    """
    Save the model to a file using pickle.
    """
    os.makedirs("models", exist_ok=True)
    filepath = os.path.join("models", fname)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"[+] Model saved as {filepath}")

def evaluate(model_name, val_df):
    """Load and evaluate the specified model with memory and time profiling."""
    print(f"[+] Evaluating {model_name}...")
    try:
        # Load the model
        with open(f"models/{model_name}.pkl", 'rb') as f:
            model = pickle.load(f)
        print(f"[+] Loaded model {model_name} successfully")

        # Debug: Check if the model is not empty
        if model is None:
            print(f"[!] Model {model_name} is None.")
            return 0, 0, 0

        # Profile memory and time
        start_time = time.time()
        mem_usage = memory_usage((test_model, (model, val_df)), interval=0.1, timeout=None)
        elapsed_time = time.time() - start_time

        # Get evaluation metrics
        correct_predictions, total_cases, accuracy = test_model(model, val_df)
        avg_memory = max(mem_usage) - min(mem_usage)

        print(f"[+] {model_name} - Time: {elapsed_time:.2f}s, Memory: {avg_memory:.2f} MB")
        print(f"  Total Test Cases: {total_cases}")
        print(f"  Total Correct Predictions: {correct_predictions}")
        print(f"  Accuracy: {accuracy:.2f}%")

        return accuracy, elapsed_time, avg_memory
    except FileNotFoundError:
        print(f"[!] {model_name}.pkl not found!")
        return 0, 0, 0


############
## Driver ##
############

def main():
    # Load data
    train_df, val_df = load_data()

    if train_df is None or val_df is None:
        print("[!] Data loading failed, exiting.")
        return

    # Task 1: Create and save Base Model
    print("[+] Creating and saving Base Model...")
    base_model = make_base_network(train_df.copy())
    save_model("base_model.pkl", base_model)
    plot_network(base_model, title="Fully Connected Base Model")

    # Task 2: Create and save Pruned Model
    print("[+] Creating and saving Pruned Model...")
    pruned_model = make_pruned_network(train_df.copy())
    save_model("pruned_model.pkl", pruned_model)
    plot_network(pruned_model, title="Pruned Model")

    # Task 3: Create and save Optimized Model
    print("[+] Creating and saving Optimized Model...")
    optimized_model = make_optimized_network(train_df.copy())
    save_model("optimized_model.pkl", optimized_model)
    plot_network(optimized_model, title="Optimized Model")

    # Evaluate the models
    print("\n[+] Evaluating Models...")
    base_accuracy, base_time, base_mem = evaluate("base_model", val_df)
    pruned_accuracy, pruned_time, pruned_mem = evaluate("pruned_model", val_df)
    optimized_accuracy, optimized_time, optimized_mem = evaluate("optimized_model", val_df)

    # Summary of results
    print("\n[+] Summary:")
    print(f"Base Model - Accuracy: {base_accuracy:.2f}%, Time: {base_time:.2f}s, Memory: {base_mem:.2f} MB")
    print(f"Pruned Model - Accuracy: {pruned_accuracy:.2f}%, Time: {pruned_time:.2f}s, Memory: {pruned_mem:.2f} MB")
    print(f"Optimized Model - Accuracy: {optimized_accuracy:.2f}%, Time: {optimized_time:.2f}s, Memory: {optimized_mem:.2f} MB")
    print("[+] Done")

if __name__ == "__main__":
    main()
