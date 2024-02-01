import os
import random
from collections import defaultdict
import argparse

''' file I/O functions '''
def write_lines_to_file(lines, filepath):
    with open(filepath, 'w') as file:
        for line in lines:
            file.write(f"{line}\n")

def read_nodes_from_file(filepath):
    nodes = set()
    with open(filepath, 'r') as file:
        for line in file.readlines():
            node1, _, node2 = line.strip().split('\t')
            nodes.add(node1)
            nodes.add(node2)
    return nodes

def collect_nodes(lines):
    nodes = set()
    for line in lines:
        node1, _, node2 = line.strip().split('\t')
        nodes.add(node1)
        nodes.add(node2)
    return list(nodes)

def delete_test_data_files(split_style):
    test_data_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/{split_style}')
    
    if not os.path.exists(test_data_dir):
        print(f"The directory {test_data_dir} does not exist.")
        return

    for filename in os.listdir(test_data_dir):
        file_path = os.path.join(test_data_dir, filename)
        try:
            if os.path.isfile(file_path):
                if filename != 'ad_pre.txt':  # Skip deleting 'ad_pre.txt'
                    os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

def read_nodes_and_edges_from_file(filepath):
    """Read edges from a file and return both edges and nodes."""
    with open(filepath, 'r') as file:
        lines = [line.strip() for line in file.readlines()]
    nodes = set()
    for line in lines:
        node1, node2 = line.split('\t')[:2]
        nodes.add(node1)
        nodes.add(node2)
    return nodes, set(lines)

''' split function '''
def split_kg(kg_filepath, split_style):
    try:
        # Step 1: Read the file
        file_path = os.path.join(os.path.dirname(__file__), kg_filepath)
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        total_edges = len(lines)
        print(f"Total edges in the graph: {total_edges}")

        # Step 2: Identify all unique nodes
        all_nodes = set()
        for line in lines:
            node1, _, node2 = line.split('\t')
            all_nodes.add(node1)
            all_nodes.add(node2)

        print(f"Total unique nodes: {len(all_nodes)}")
        random.seed(42)

        # Set desired ratios for Test and Valid sets
        test_ratio = 0.2
        valid_ratio = 0.2
        valid_count = int(total_edges * valid_ratio)
        test_count = int(total_edges * test_ratio)
        # The remaining will automatically belong to the training set after allocating for test and valid
        train_count = total_edges - valid_count - test_count

        # Step 3: Splitting the data based on the selected style 
        if split_style == 'fully_inductive':
            train_lines, valid_lines, test_lines = split_fully_inductive(kg_filepath)
            print(len(train_lines), len(valid_lines), len(test_lines))
    
        elif split_style == 'semi_inductive':
            train_lines, valid_lines, test_lines = split_semi_inductive(kg_filepath)

        elif split_style == 'transductive':
            # Transductive setup
            train_lines = set()
            nodes_in_train = set()
            
            for line in lines:
                node1, _, node2 = line.split('\t')
                if node1 not in nodes_in_train or node2 not in nodes_in_train:
                    train_lines.add(line)
                    nodes_in_train.add(node1)
                    nodes_in_train.add(node2)
            
            remaining_lines = set(lines) - train_lines
            remaining_lines = list(remaining_lines)  # Convert to list for random shuffling
            random.shuffle(remaining_lines)
            
            total_remaining = len(remaining_lines)
            valid_count = int(total_remaining * valid_ratio)
            test_count = int(total_remaining * test_ratio)
            
            valid_lines = remaining_lines[:valid_count]
            test_lines = remaining_lines[valid_count:(valid_count + test_count)]
            train_lines.update(remaining_lines[(valid_count + test_count):])  # Add the rest to the training set

        else:
            raise ValueError("Invalid split style. Choose 'transductive', 'semi_inductive' or 'fully_inductive'.")

        # Step 4: Report the ratios
        train_ratio = len(train_lines) / total_edges * 100
        valid_ratio = len(valid_lines) / total_edges * 100
        test_ratio = len(test_lines) / total_edges * 100
        
        print(f"Train:Test:Valid ratio = {train_ratio:.2f}%:{test_ratio:.2f}%:{valid_ratio:.2f}\n===================================")
        print(len(train_lines), len(valid_lines), len(test_lines))
        
        # Step 5: Write to files
        base_output_dir = os.path.join(os.path.dirname(__file__), f'data/GPKG/{split_style}')
        if not os.path.exists(base_output_dir):
            os.makedirs(base_output_dir)
        
        write_lines_to_file(train_lines, os.path.join(base_output_dir, 'train.txt'))
        write_lines_to_file(valid_lines, os.path.join(base_output_dir, 'valid.txt'))
        write_lines_to_file(test_lines, os.path.join(base_output_dir, 'test.txt'))

    except FileNotFoundError:
        print(f"File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

''' after the initial split, calculates the actual ratio of edges in each set. If the ratios are not close to your target split ratio, slightly adjusts the nodes allocated to each set and recalculate, iteratively. '''
def adjust_nodes_for_edge_ratios(total_nodes, target_train_ratio, target_valid_ratio, lines, max_iterations=10000):
    random.shuffle(total_nodes)
    for _ in range(max_iterations):
        num_train = int(len(total_nodes) * target_train_ratio)
        num_valid = int(len(total_nodes) * target_valid_ratio)

        train_nodes = set(total_nodes[:num_train])
        valid_nodes = set(total_nodes[num_train:num_train + num_valid])
        test_nodes = set(total_nodes[num_train + num_valid:])

        # Calculate edge counts for each split
        train_edges = sum(1 for line in lines if line.split('\t')[0] in train_nodes and line.split('\t')[1] in train_nodes)
        valid_edges = sum(1 for line in lines if line.split('\t')[0] in valid_nodes and line.split('\t')[1] in valid_nodes)
        test_edges = sum(1 for line in lines if line.split('\t')[0] in test_nodes and line.split('\t')[1] in test_nodes)
        total_edges = train_edges + valid_edges + test_edges

        # Calculate current ratios
        current_train_ratio = train_edges / total_edges
        current_valid_ratio = valid_edges / total_edges

        # Check if current ratios are close enough to target ratios
        if abs(current_train_ratio - target_train_ratio) <= 0.05 and abs(current_valid_ratio - target_valid_ratio) <= 0.05:
            return train_nodes, valid_nodes, test_nodes

        # Adjust total_nodes by shuffling for the next iteration if not close enough
        random.shuffle(total_nodes)

    # Return the last attempt if max_iterations reached without achieving target ratios
    return train_nodes, valid_nodes, test_nodes

''' fully inductive split '''
def split_fully_inductive(kg_filepath):
    try:
        file_path = os.path.join(os.path.dirname(__file__), kg_filepath)
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        all_nodes = set()
        for line in lines:
            node1, node2 = line.split('\t')[:2]
            all_nodes.add(node1)
            all_nodes.add(node2)

        random.seed(42)
        total_nodes = list(all_nodes)
        
        # Target ratios
        target_train_ratio = 0.6
        target_valid_ratio = 0.2

        # Attempt to adjust nodes to get closer to target edge ratios
        train_nodes, valid_nodes, test_nodes = adjust_nodes_for_edge_ratios(
            total_nodes, target_train_ratio, target_valid_ratio, lines
        )

        # Extract lines based on final node sets
        train_lines = [line for line in lines if line.split('\t')[0] in train_nodes and line.split('\t')[1] in train_nodes]
        valid_lines = [line for line in lines if line.split('\t')[0] in valid_nodes and line.split('\t')[1] in valid_nodes]
        test_lines = [line for line in lines if line.split('\t')[0] in test_nodes and line.split('\t')[1] in test_nodes]

        return train_lines, valid_lines, test_lines

    except FileNotFoundError:
        print(f"File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

''' semi inductive split '''
def split_semi_inductive(kg_filepath):
    try:
        file_path = os.path.join(os.path.dirname(__file__), kg_filepath)
        with open(file_path, 'r') as file:
            lines = [line.strip() for line in file.readlines()]

        # Shuffle for randomness
        random.seed(42)  # Ensuring reproducibility
        random.shuffle(lines)

        total_edges = len(lines)
        test_ratio = 0.2
        valid_ratio = 0.2

        # Calculate counts for each set
        valid_count = int(total_edges * valid_ratio)
        test_count = int(total_edges * test_ratio)
        train_count = total_edges - valid_count - test_count

        # Initialize sets
        train_lines = []
        valid_lines = []
        test_lines = []

        all_nodes, train_nodes = set(), set()

        # First, allocate training edges to ensure all nodes are seen during training
        for line in lines[:train_count]:
            node1, _, node2 = line.split('\t')
            train_lines.append(line)
            train_nodes.add(node1)
            train_nodes.add(node2)
            all_nodes.add(node1)
            all_nodes.add(node2)

        # Allocate validation and test edges
        for line in lines[train_count:]:
            node1, _, node2 = line.split('\t')
            all_nodes.add(node1)
            all_nodes.add(node2)
            # Prioritize filling the validation set while including nodes from training
            if len(valid_lines) < valid_count and (node1 in train_nodes or node2 in train_nodes):
                valid_lines.append(line)
            elif len(test_lines) < test_count and (node1 in train_nodes or node2 in train_nodes):
                test_lines.append(line)
            # Optionally, further logic to handle leftover edges

        # Ensure every node is at least in the training set
        missing_nodes = all_nodes - train_nodes
        # Logic to handle missing_nodes if necessary, for now, we assume minimal or no missing nodes for semi-inductive

        return train_lines, valid_lines, test_lines

    except FileNotFoundError:
        print(f"File not found at {kg_filepath}")
        return [], [], []  # Return empty lists in case of error
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], [], []  # Return empty lists in case of error


''' testing functions (testing whether all nodes in the validation and test sets also appear in the training set) '''
def test_splits_transductive(train_filepath, valid_filepath, test_filepath):
    # Read nodes from each file
    train_nodes = read_nodes_from_file(train_filepath)
    valid_nodes = read_nodes_from_file(valid_filepath)
    test_nodes = read_nodes_from_file(test_filepath)

    # Check if all nodes in valid and test sets are also in the train set
    valid_diff = valid_nodes.difference(train_nodes)
    test_diff = test_nodes.difference(train_nodes)

    if len(valid_diff) == 0 and len(test_diff) == 0:
        print("✔ Test Passed: All nodes in the validation and test sets also appear in the training set.")
    else:
        print("✘ Test Failed")
        if len(valid_diff) > 0:
            print(f"Nodes in validation set not found in training set: {len(valid_diff)}")
        if len(test_diff) > 0:
            print(f"Nodes in test set not found in training set: {len(test_diff)}")

'''Includes checks for:
- New nodes in both validation and test sets.
- Additional new nodes in the test set that are not in the validation set.
- The ratio of new nodes in the test set being larger than in the validation set.'''
def test_splits_semi_inductive(train_filepath, valid_filepath, test_filepath):
    
    # Read nodes from each file
    train_nodes = read_nodes_from_file(train_filepath)
    valid_nodes = read_nodes_from_file(valid_filepath)
    test_nodes = read_nodes_from_file(test_filepath)

    # Identify new nodes in validation and test sets
    valid_new_nodes = valid_nodes.difference(train_nodes)
    test_new_nodes = test_nodes.difference(train_nodes)

    # Additional nodes in test set not in validation set
    additional_test_nodes = test_new_nodes.difference(valid_new_nodes)

    if valid_new_nodes or test_new_nodes:
        print("✔ Test Passed: There are new nodes in validation and/or test sets.")
        if valid_new_nodes:
            print(f"New nodes in validation set: {len(valid_new_nodes)}")
        if test_new_nodes:
            print(f"New nodes in test set: {len(test_new_nodes)}")
        if additional_test_nodes:
            print(f"Additional new nodes in test set that are not in validation set: {len(additional_test_nodes)}")
        if len(test_new_nodes) > len(valid_new_nodes):
            print("✔ Test Passed: The number of new nodes in the test set is larger than in the validation set.")
        else:
            print("✘ Test Failed: The number of new nodes in the test set is not larger than in the validation set.")
    else:
        print("✘ Test Failed: No new nodes in validation and test sets. This does not align with semi-inductive setup.")

def test_splits_fully_inductive(train_filepath, valid_filepath, test_filepath):
    train_nodes, train_edges = read_nodes_and_edges_from_file(train_filepath)
    valid_nodes, valid_edges = read_nodes_and_edges_from_file(valid_filepath)
    test_nodes, test_edges = read_nodes_and_edges_from_file(test_filepath)

    # Test 1: Disjoint Node Sets
    if train_nodes.isdisjoint(valid_nodes) and train_nodes.isdisjoint(test_nodes) and valid_nodes.isdisjoint(test_nodes):
        print("✔ Node sets are disjoint.")
    else:
        print("✘ Node sets are not disjoint.")

    # Test 2: Correct Edge Allocation
    correct_train_allocation = all(node1 in train_nodes and node2 in train_nodes for node1, node2 in (edge.split('\t')[:2] for edge in train_edges))
    correct_valid_allocation = all(node1 in valid_nodes and node2 in valid_nodes for node1, node2 in (edge.split('\t')[:2] for edge in valid_edges))
    correct_test_allocation = all(node1 in test_nodes and node2 in test_nodes for node1, node2 in (edge.split('\t')[:2] for edge in test_edges))
    
    if correct_train_allocation and correct_valid_allocation and correct_test_allocation:
        print("✔ All edges are correctly allocated.")
    else:
        print("✘ Some edges are incorrectly allocated.")

    # Test 3: Non-Empty Sets (Optional, based on requirements)
    if train_edges and valid_edges and test_edges:
        print("✔ All sets contain edges.")
    else:
        print("✘ One or more sets are empty.")


def run_test(split_style):
    base_dir = f'data/GPKG/{split_style}'  # Replace with the path where your txt files are stored
    train_filepath = os.path.join(base_dir, 'train.txt')
    valid_filepath = os.path.join(base_dir, 'valid.txt')
    test_filepath = os.path.join(base_dir, 'test.txt')

    if split_style == 'transductive':
        test_splits_transductive(train_filepath, valid_filepath, test_filepath)
    elif split_style == 'semi_inductive':
        test_splits_semi_inductive(train_filepath, valid_filepath, test_filepath)
    elif split_style == 'fully_inductive':
        test_splits_fully_inductive(train_filepath, valid_filepath, test_filepath)

''' main '''
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split a knowledge graph into training, validation, and test sets.")
    parser.add_argument('-kg_filepath', type=str, default=None, help="Path to the knowledge graph file.")
    parser.add_argument('-split_style', type=str, default='transductive', help="Split style: transductive, semi_inductive, or fully_inductive?")

    args = parser.parse_args()
    kg_filepath = args.kg_filepath
    split_style = args.split_style

    delete_test_data_files(split_style)

    if kg_filepath:
        print(f'Splitting {kg_filepath} in {split_style} setup...\n===================================')
        split_kg(kg_filepath, split_style)
    else:
        split_kg()

    run_test(split_style)
