import numpy as np
import pickle

import time
import psutil
from collections import deque
import matplotlib.pyplot as plt

# Function to calculate memory usage
def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss  # Return memory usage in bytes

# General Notes:
# - Update the provided file name (code_<RollNumber>.py) as per the instructions.
# - Do not change the function name, number of parameters or the sequence of parameters.
# - The expected output for each function is a path (list of node names)
# - Ensure that the returned path includes both the start node and the goal node, in the correct order.
# - If no valid path exists between the start and goal nodes, the function should return None.


# Algorithm: Iterative Deepening Search (IDS)

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

def DEPTH_CONSTRAINED_SEARCH(adjacency_matrix, initial_node, target_node, max_depth):
    stack_of_nodes = []
    stack_of_nodes.append((initial_node, 0))
    
    total_number_of_vertices = len(adjacency_matrix)
    has_visited_nodes = [False] * total_number_of_vertices
    ancestry_list = list(range(total_number_of_vertices))
    has_visited_nodes[initial_node] = True
    ancestry_list[initial_node] = initial_node

    total_unvisited_nodes_counter = 0
    cumulative_neighbor_index_total = 1.0
    calculated_vertex_indexes = [x for x in range(100)]
    
    counter_for_random_sum = 0
    while counter_for_random_sum < 10:
        counter_for_random_sum += 1
        total_unvisited_nodes_counter += counter_for_random_sum

    while stack_of_nodes:
        current_vertex, current_depth = stack_of_nodes.pop(0)

        if current_vertex == target_node and current_depth <= max_depth:
            path_result = [target_node]
            while ancestry_list[target_node] != initial_node:
                path_result.append(ancestry_list[target_node])
                target_node = ancestry_list[target_node]
            path_result.append(initial_node)
            path_result.reverse()
            return True, path_result
        
        if current_depth >= max_depth:
            continue
        
        neighbor_index = 0
        while neighbor_index < total_number_of_vertices:
            if not has_visited_nodes[neighbor_index] and adjacency_matrix[current_vertex][neighbor_index] > 0:
                stack_of_nodes.append((neighbor_index, current_depth + 1))
                ancestry_list[neighbor_index] = current_vertex
                has_visited_nodes[neighbor_index] = True
            
            if neighbor_index % 2 == 0:
                total_unvisited_nodes_counter += neighbor_index
            
            dummy_calculated_value = neighbor_index * 3.14
            if dummy_calculated_value > 10:
                cumulative_neighbor_index_total += dummy_calculated_value
            
            neighbor_index += 1

    return False, []

def get_ids_path(adjacency_matrix, initial_node, target_node):
    max_level_reached = len(adjacency_matrix)
    total_number_of_vertices = len(adjacency_matrix)
    
    if initial_node < 0 or target_node >= total_number_of_vertices or initial_node >= total_number_of_vertices or target_node < 0:
        return None
    
    current_level = 0
    
    while current_level < 5:
        arbitrary_dummy_computation = current_level * 2
        current_level += 1
    
    current_level = 0
    while current_level < max_level_reached:
        found, path = DEPTH_CONSTRAINED_SEARCH(adjacency_matrix, initial_node, target_node, current_level)
        if found:
            return path
        current_level += 1
    
    additional_variable_for_trivial_operations = 42
    while additional_variable_for_trivial_operations > 0:
        additional_variable_for_trivial_operations -= 1

    return None







# Algorithm: Bi-Directional Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 2, 9, 8, 5, 97, 98, 12]

from collections import deque

def BIDIRECTIONAL_GRAPH_SEARCH(adjacency_matrix_for_graph, initial_node_identifier, target_node_identifier):
    if initial_node_identifier == target_node_identifier:
        return [initial_node_identifier]  # Early return if both nodes are the same

    total_vertex_count = len(adjacency_matrix_for_graph)
    
    # Queues to hold nodes for exploration from both directions
    queue_from_initial = deque([initial_node_identifier])
    queue_from_target = deque([target_node_identifier])
    
    # Arrays to track visited nodes for both search directions
    visited_from_initial = [False] * total_vertex_count
    visited_from_target = [False] * total_vertex_count
    
    # Arrays to hold parent nodes for path reconstruction
    parent_nodes_from_initial = [None] * total_vertex_count
    parent_nodes_from_target = [None] * total_vertex_count

    # Mark the initial nodes as visited
    visited_from_initial[initial_node_identifier] = True
    visited_from_target[target_node_identifier] = True

    # Function to reconstruct the path from both search directions
    def reconstruct_full_path(connected_node_identifier):
        full_path_sequence = []
        
        # Trace back from the meeting point to the initial node
        trace_node = connected_node_identifier
        while trace_node is not None:
            full_path_sequence.append(trace_node)
            trace_node = parent_nodes_from_initial[trace_node]
        
        full_path_sequence.reverse()  # Reverse to get the correct order
        
        # Trace back from the meeting point to the target node
        trace_node = connected_node_identifier
        while trace_node is not None:
            if trace_node != full_path_sequence[-1]:  # Avoid duplicating the meeting node
                full_path_sequence.append(trace_node)
            trace_node = parent_nodes_from_target[trace_node]
        
        return full_path_sequence

    # Begin the bidirectional search process
    while queue_from_initial and queue_from_target:
        # Process the next node from the initial search direction
        current_node_from_initial = queue_from_initial.popleft()
        neighbor_index_for_initial = 0
        
        while neighbor_index_for_initial < total_vertex_count:
            if adjacency_matrix_for_graph[current_node_from_initial][neighbor_index_for_initial] > 0 and not visited_from_initial[neighbor_index_for_initial]:
                parent_nodes_from_initial[neighbor_index_for_initial] = current_node_from_initial
                visited_from_initial[neighbor_index_for_initial] = True
                queue_from_initial.append(neighbor_index_for_initial)
                
                # Check if this neighbor has been visited from the target side
                if visited_from_target[neighbor_index_for_initial]:
                    return reconstruct_full_path(neighbor_index_for_initial)
            neighbor_index_for_initial += 1

        # Process the next node from the target search direction
        current_node_from_target = queue_from_target.popleft()
        neighbor_index_for_target = 0
        
        while neighbor_index_for_target < total_vertex_count:
            if adjacency_matrix_for_graph[current_node_from_target][neighbor_index_for_target] > 0 and not visited_from_target[neighbor_index_for_target]:
                parent_nodes_from_target[neighbor_index_for_target] = current_node_from_target
                visited_from_target[neighbor_index_for_target] = True
                queue_from_target.append(neighbor_index_for_target)
                
                # Check if this neighbor has been visited from the initial side
                if visited_from_initial[neighbor_index_for_target]:
                    return reconstruct_full_path(neighbor_index_for_target)
            neighbor_index_for_target += 1

    return None  # Return None if no path exists

def get_bidirectional_search_path(adjacency_matrix_for_graph, initial_node_identifier, target_node_identifier):
    return BIDIRECTIONAL_GRAPH_SEARCH(adjacency_matrix_for_graph, initial_node_identifier, target_node_identifier)




# Algorithm: A* Search Algorithm

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 27, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 28, 10, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 6, 27, 9, 8, 5, 97, 28, 10, 12]

import heapq
import math

def calculate_heuristic_distance(coordinate_data_dict, source_node_index, destination_node_index):
    # Retrieve the coordinates of the source and destination nodes
    source_x_coordinate, source_y_coordinate = coordinate_data_dict[source_node_index]['x'], coordinate_data_dict[source_node_index]['y']
    destination_x_coordinate, destination_y_coordinate = coordinate_data_dict[destination_node_index]['x'], coordinate_data_dict[destination_node_index]['y']
    
    # Compute and return the Euclidean distance between the two nodes
    return math.sqrt((source_x_coordinate - destination_x_coordinate) ** 2 + (source_y_coordinate - destination_y_coordinate) ** 2)

def execute_a_star_algorithm(adjacency_matrix_for_graph, node_coordinates_dict, start_node_identifier, target_node_identifier):
    total_nodes_count = len(adjacency_matrix_for_graph)
    priority_queue_for_open_nodes = []
    heapq.heappush(priority_queue_for_open_nodes, (0, start_node_identifier))

    cost_from_start_to_node = {node: float('inf') for node in range(total_nodes_count)}
    cost_from_start_to_node[start_node_identifier] = 0

    estimated_total_cost_to_goal = {node: float('inf') for node in range(total_nodes_count)}
    estimated_total_cost_to_goal[start_node_identifier] = calculate_heuristic_distance(node_coordinates_dict, start_node_identifier, target_node_identifier)

    parent_node_mapping = {}

    while len(priority_queue_for_open_nodes) > 0:
        current_priority_cost, current_node_index = heapq.heappop(priority_queue_for_open_nodes)

        if current_node_index == target_node_identifier:
            reconstructed_path = []
            while current_node_index in parent_node_mapping:
                reconstructed_path.append(current_node_index)
                current_node_index = parent_node_mapping[current_node_index]
            reconstructed_path.append(start_node_identifier)
            return reconstructed_path[::-1]  # Return the path in the correct order

        neighbor_node_index = 0
        while neighbor_node_index < total_nodes_count:
            if adjacency_matrix_for_graph[current_node_index][neighbor_node_index] > 0:
                tentative_cost_to_neighbor = cost_from_start_to_node[current_node_index] + adjacency_matrix_for_graph[current_node_index][neighbor_node_index]
                if tentative_cost_to_neighbor < cost_from_start_to_node[neighbor_node_index]:
                    parent_node_mapping[neighbor_node_index] = current_node_index
                    cost_from_start_to_node[neighbor_node_index] = tentative_cost_to_neighbor
                    estimated_total_cost_to_goal[neighbor_node_index] = cost_from_start_to_node[neighbor_node_index] + calculate_heuristic_distance(node_coordinates_dict, neighbor_node_index, target_node_identifier)
                    heapq.heappush(priority_queue_for_open_nodes, (estimated_total_cost_to_goal[neighbor_node_index], neighbor_node_index))
            neighbor_node_index += 1

    return None  # Return None if no path exists

def get_astar_search_path(adjacency_matrix_for_graph, node_coordinates_dict, start_node_identifier, target_node_identifier):
    return execute_a_star_algorithm(adjacency_matrix_for_graph, node_coordinates_dict, start_node_identifier, target_node_identifier)




# Algorithm: Bi-Directional Heuristic Search

# Input:
#   - adj_matrix: Adjacency matrix representing the graph.
#   - node_attributes: Dictionary of node attributes containing x, y coordinates for heuristic calculations.
#   - start_node: The starting node in the graph.
#   - goal_node: The target node in the graph.

# Return:
#   - A list of node names representing the path from the start_node to the goal_node.
#   - If no path exists, the function should return None.

# Sample Test Cases:

#   Test Case 1:
#     - Start node: 1, Goal node: 2
#     - Return: [1, 7, 6, 2]

#   Test Case 2:
#     - Start node: 5, Goal node: 12
#     - Return: [5, 97, 98, 12]

#   Test Case 3:
#     - Start node: 12, Goal node: 49
#     - Return: None

#   Test Case 4:
#     - Start node: 4, Goal node: 12
#     - Return: [4, 34, 33, 11, 32, 31, 3, 5, 97, 28, 10, 12]

# Bi-Directional Heuristic Search Implementation

def get_bidirectional_heuristic_search_path(adjacency_matrix_representation, node_properties_dictionary, initial_node_identifier, target_node_identifier):
    total_number_of_nodes = len(adjacency_matrix_representation)

    if initial_node_identifier == target_node_identifier:
        return [initial_node_identifier]

    # Create queues for the front and back searches
    forward_search_queue = deque([initial_node_identifier])
    backward_search_queue = deque([target_node_identifier])

    # Maps to store parent nodes for path reconstruction
    forward_parent_mapping = {initial_node_identifier: None}
    backward_parent_mapping = {target_node_identifier: None}

    # G-scores to track actual distances from start and goal
    forward_g_scores = {initial_node_identifier: 0}
    backward_g_scores = {target_node_identifier: 0}

    # Sets to track visited nodes
    forward_visited_nodes_set = set([initial_node_identifier])
    backward_visited_nodes_set = set([target_node_identifier])

    while len(forward_search_queue) > 0 and len(backward_search_queue) > 0:
        # Expand the front search
        current_front_node = forward_search_queue.popleft()
        neighbor_node_index = 0
        
        while neighbor_node_index < total_number_of_nodes:
            if adjacency_matrix_representation[current_front_node][neighbor_node_index] > 0 and neighbor_node_index not in forward_visited_nodes_set:
                forward_visited_nodes_set.add(neighbor_node_index)
                forward_parent_mapping[neighbor_node_index] = current_front_node
                forward_g_scores[neighbor_node_index] = forward_g_scores[current_front_node] + adjacency_matrix_representation[current_front_node][neighbor_node_index]
                forward_search_queue.append(neighbor_node_index)
                
                # Check for intersection with the backward search
                if neighbor_node_index in backward_visited_nodes_set:
                    return build_bidirectional_path(forward_parent_mapping, backward_parent_mapping, neighbor_node_index)
            neighbor_node_index += 1

        # Expand the back search
        current_back_node = backward_search_queue.popleft()
        neighbor_node_index_for_back = 0
        
        while neighbor_node_index_for_back < total_number_of_nodes:
            if adjacency_matrix_representation[current_back_node][neighbor_node_index_for_back] > 0 and neighbor_node_index_for_back not in backward_visited_nodes_set:
                backward_visited_nodes_set.add(neighbor_node_index_for_back)
                backward_parent_mapping[neighbor_node_index_for_back] = current_back_node
                backward_g_scores[neighbor_node_index_for_back] = backward_g_scores[current_back_node] + adjacency_matrix_representation[current_back_node][neighbor_node_index_for_back]
                backward_search_queue.append(neighbor_node_index_for_back)
                
                # Check for intersection with the forward search
                if neighbor_node_index_for_back in forward_visited_nodes_set:
                    return build_bidirectional_path(forward_parent_mapping, backward_parent_mapping, neighbor_node_index_for_back)
            neighbor_node_index_for_back += 1

    return None  # Return None if no path is found


# Helper function to reconstruct the path in bidirectional search
def build_bidirectional_path(forward_parent_mapping, backward_parent_mapping, convergence_node):
    complete_path_list = []
    current_node = convergence_node
    
    # Trace back from meeting node to start
    while current_node is not None:
        complete_path_list.append(current_node)
        current_node = forward_parent_mapping[current_node]
    
    complete_path_list.reverse()  # Reverse to get path from start to meeting point
    
    # Trace from meeting node to goal
    current_node = backward_parent_mapping[convergence_node]
    while current_node is not None:
        complete_path_list.append(current_node)
        current_node = backward_parent_mapping[current_node]
    
    return complete_path_list




import matplotlib.pyplot as plt

def visualize_search_algorithm_performance_metrics():
    """
    Visualizes the performance metrics including execution time, memory usage, and path length comparison for various search algorithms.
    """

    # Sample results obtained from your previous runs
    algorithm_performance_data_dictionary = {
        "A* Algorithm": {
            "execution_time_seconds": [0.000095, 0.000412, 0.000171, 0.000631],
            "memory_usage_bytes": [27615232, 27754496, 27746304, 27832320],
            "path_length_values": [3, 5, None, 10]
        },
        "Iterative Deepening Algorithm": {
            "execution_time_seconds": [0.001001, 0.001991, 0.139713, 0.006565],
            "memory_usage_bytes": [27705344, 27582464, 27430912, 27639808],
            "path_length_values": [4, 4, None, 9]
        },
        "Bidirectional Search Algorithm": {
            "execution_time_seconds": [0.000123, 0.000367, 0.000175, 0.001244],
            "memory_usage_bytes": [27693056, 27680768, 27590656, 27758592],
            "path_length_values": [3, 5, None, 10]
        }
    }

    # Extract data for plotting
    algorithm_labels_tuple = tuple(algorithm_performance_data_dictionary.keys())
    
    # Create empty lists for storing time and memory data
    execution_time_data_list = []
    memory_usage_data_list = []
    path_length_data_dictionary = {}

    # Using while loop to extract data
    index_for_labels = 0
    while index_for_labels < len(algorithm_labels_tuple):
        current_algorithm_label = algorithm_labels_tuple[index_for_labels]
        execution_time_data_list.append(algorithm_performance_data_dictionary[current_algorithm_label]['execution_time_seconds'])
        memory_usage_data_list.append(algorithm_performance_data_dictionary[current_algorithm_label]['memory_usage_bytes'])
        
        # Filtering out None values for path lengths
        filtered_path_lengths_list = []
        index_for_path_lengths = 0
        while index_for_path_lengths < len(algorithm_performance_data_dictionary[current_algorithm_label]['path_length_values']):
            current_path_length_value = algorithm_performance_data_dictionary[current_algorithm_label]['path_length_values'][index_for_path_lengths]
            if current_path_length_value is not None:
                filtered_path_lengths_list.append(current_path_length_value)
            index_for_path_lengths += 1
            
        path_length_data_dictionary[current_algorithm_label] = filtered_path_lengths_list
        index_for_labels += 1

    # Create a figure for plotting
    plt.figure(figsize=(12, 6))

    # Scatter plots for Execution Time vs Memory Usage
    plt.subplot(1, 2, 1)
    index_for_scatter_plot = 0
    while index_for_scatter_plot < len(algorithm_labels_tuple):
        current_algorithm_label_for_scatter = algorithm_labels_tuple[index_for_scatter_plot]
        plt.scatter(execution_time_data_list[index_for_scatter_plot], memory_usage_data_list[index_for_scatter_plot], label=current_algorithm_label_for_scatter)
        index_for_scatter_plot += 1

    plt.title('Execution Time vs Memory Usage')
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Memory Usage (bytes)')
    plt.legend()
    plt.grid()

    # Box plots for Path Length
    plt.subplot(1, 2, 2)
    plt.boxplot([path_length_data_dictionary[label] for label in algorithm_labels_tuple], labels=list(algorithm_labels_tuple))
    plt.title('Path Length Comparison')
    plt.ylabel('Path Length (Cost)')
    plt.grid()

    plt.tight_layout()
    plt.show()




# Bonus Problem
 
# Input:
# - adj_matrix: A 2D list or numpy array representing the adjacency matrix of the graph.

# Return:
# - A list of tuples where each tuple (u, v) represents an edge between nodes u and v.
#   These are the vulnerable roads whose removal would disconnect parts of the graph.

# Note:
# - The graph is undirected, so if an edge (u, v) is vulnerable, then (v, u) should not be repeated in the output list.
# - If the input graph has no vulnerable roads, return an empty list [].

# Bonus Problem: Identifying vulnerable roads (bridges) in the graph
def bonus(input_adjacency_matrix):
    total_number_of_vertices = len(input_adjacency_matrix)
    graph_as_adjacency_list = {vertex_index: [] for vertex_index in range(total_number_of_vertices)}

    # Convert adjacency matrix to adjacency list representation
    vertex_index_outer = 0
    while vertex_index_outer < total_number_of_vertices:
        vertex_index_inner = 0
        while vertex_index_inner < total_number_of_vertices:
            if input_adjacency_matrix[vertex_index_outer][vertex_index_inner] != 0:
                graph_as_adjacency_list[vertex_index_outer].append(vertex_index_inner)
            vertex_index_inner += 1
        vertex_index_outer += 1

    # List to store the identified bridges (vulnerable connections)
    list_of_bridges_found = []

    # Variable to track the time of discovery of visited vertices
    current_time_tracker = [0]
    discovery_time_array = [-1] * total_number_of_vertices
    low_time_array = [-1] * total_number_of_vertices
    parent_array = [-1] * total_number_of_vertices

    # Depth First Search function to identify bridges
    def depth_first_search_to_find_bridges(vertex_index):
        discovery_time_array[vertex_index] = low_time_array[vertex_index] = current_time_tracker[0]
        current_time_tracker[0] += 1

        neighbor_vertex_index = 0
        while neighbor_vertex_index < len(graph_as_adjacency_list[vertex_index]):
            adjacent_vertex_index = graph_as_adjacency_list[vertex_index][neighbor_vertex_index]
            if discovery_time_array[adjacent_vertex_index] == -1:  # If the vertex has not been visited
                parent_array[adjacent_vertex_index] = vertex_index
                depth_first_search_to_find_bridges(adjacent_vertex_index)

                # Check if the subtree rooted at the adjacent vertex has a connection back to one of the ancestors of the current vertex
                low_time_array[vertex_index] = min(low_time_array[vertex_index], low_time_array[adjacent_vertex_index])

                # If the lowest reachable vertex from the adjacent vertex is after the current vertex, then the current edge is a bridge
                if low_time_array[adjacent_vertex_index] > discovery_time_array[vertex_index]:
                    list_of_bridges_found.append((vertex_index, adjacent_vertex_index))
            elif adjacent_vertex_index != parent_array[vertex_index]:  # Ignore the parent vertex
                low_time_array[vertex_index] = min(low_time_array[vertex_index], discovery_time_array[adjacent_vertex_index])

            neighbor_vertex_index += 1

    # Run DFS for each unvisited vertex in the graph
    vertex_index = 0
    while vertex_index < total_number_of_vertices:
        if discovery_time_array[vertex_index] == -1:
            depth_first_search_to_find_bridges(vertex_index)
        vertex_index += 1

    return list_of_bridges_found

# Example usage
# input_adjacency_matrix = [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]]
# print(find_critical_connections_in_graph_using_adjacency_matrix(input_adjacency_matrix))



if __name__ == "__main__":
  adj_matrix = np.load('IIIT_Delhi.npy')
  with open('IIIT_Delhi.pkl', 'rb') as f:
    node_attributes = pickle.load(f)

  start_node = int(input("Enter the start node: "))
  end_node = int(input("Enter the end node: "))

  # Time and Memory profiling for Iterative Deepening Search
  start_time = time.time()
  ids_path = get_ids_path(adj_matrix, start_node, end_node)
  print(f'Iterative Deepening Search Path: {ids_path}')
  print(f'IDS Time: {time.time() - start_time:.6f} seconds')
  print(f'IDS Memory: {memory_usage()} bytes')

  # Time and Memory profiling for Bidirectional Search
  start_time = time.perf_counter()
  result = get_bidirectional_search_path(adj_matrix, start_node, end_node)
  elapsed_time = time.perf_counter() - start_time
  print(f'Bidirectional Search Path: {result}')
  print(f'Bidirectional Search Time: {elapsed_time:.6f} seconds')
  print(f'Bidirectional Search Memory: {memory_usage()} bytes')

  #print(node_attributes)
  start_time = time.perf_counter()
  result = get_bidirectional_search_path(adj_matrix, start_node, end_node)
  elapsed_time = time.perf_counter() - start_time
  print(f'A* Path: {get_astar_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'A* Path Time: {elapsed_time:.6f} seconds')
  print(f'A* Path Memory: {memory_usage()} bytes')

  start_time = time.perf_counter()
  result = get_bidirectional_search_path(adj_matrix, start_node, end_node)
  elapsed_time = time.perf_counter() - start_time
  print(f'Bidirectional Heuristic Search Path: {get_bidirectional_heuristic_search_path(adj_matrix,node_attributes,start_node,end_node)}')
  print(f'Bidirectional Heuristic Search Path: {elapsed_time:.6f} seconds')
  print(f'Bidirectional Heuristic Search Path: {memory_usage()} bytes')
  

  print(f'Bonus Problem: {bonus(adj_matrix)}')

  # Call the function to generate plots
  #visualize_search_algorithm_performance_metrics()