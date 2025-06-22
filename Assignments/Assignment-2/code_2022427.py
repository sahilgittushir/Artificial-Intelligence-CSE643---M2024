# Boilerplate for AI Assignment — Knowledge Representation, Reasoning and Planning
# CSE 643

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import networkx as nx
from pyDatalog import pyDatalog
from collections import defaultdict, deque

## ****IMPORTANT****
## Don't import or use any other libraries other than defined above
## Otherwise your code file will be rejected in the automated testing

# ------------------ Global Variables ------------------
route_to_stops = defaultdict(list)  # Mapping of route IDs to lists of stops
trip_to_route = {}                   # Mapping of trip IDs to route IDs
stop_trip_count = defaultdict(int)    # Count of trips for each stop
fare_rules = {}                      # Mapping of route IDs to fare information
merged_fare_df = pd.DataFrame()               # To be initialized in create_kb()

# Load static data from GTFS (General Transit Feed Specification) files
df_stops = pd.read_csv('GTFS/stops.txt')
df_routes = pd.read_csv('GTFS/routes.txt')
df_stop_times = pd.read_csv('GTFS/stop_times.txt')
df_fare_attributes = pd.read_csv('GTFS/fare_attributes.txt')
df_trips = pd.read_csv('GTFS/trips.txt')
df_fare_rules = pd.read_csv('GTFS/fare_rules.txt')

# ------------------ Function Definitions ------------------

# Function to create knowledge base from the loaded data
def create_kb():
    """
    Create the knowledge base by populating relationships between routes, trips, and stops.
    """
    global route_to_stops, trip_to_route, stop_trip_count

    # Initialize dictionaries
    route_to_stops = {}
    trip_to_route = {}
    stop_trip_count = {}

    # Map trip IDs to route IDs
    trip_index = 0
    trip_data = df_trips.values
    while trip_index < len(trip_data):
        trip_identifier = trip_data[trip_index][2]  # trip_id
        route_identifier = trip_data[trip_index][0]  # route_id
        trip_to_route[trip_identifier] = route_identifier
        trip_index += 1

    # Map route IDs to unique stops
    stop_grouping = df_stop_times.groupby('trip_id')['stop_id'].apply(list).to_dict()
    stop_keys = list(stop_grouping.keys())
    route_idx = 0
    while route_idx < len(stop_keys):
        trip_identifier = stop_keys[route_idx]
        stop_list = stop_grouping[trip_identifier]
        route_identifier = trip_to_route.get(trip_identifier)

        if route_identifier:
            if route_identifier not in route_to_stops:
                route_to_stops[route_identifier] = []
            route_to_stops[route_identifier].extend(stop_list)
        route_idx += 1

    # Ensure stops for each route are unique and preserve order
    route_keys = list(route_to_stops.keys())
    route_counter = 0
    while route_counter < len(route_keys):
        route_identifier = route_keys[route_counter]
        seen_stops = set()
        unique_stop_list = []
        stop_idx = 0
        stops_for_route = route_to_stops[route_identifier]
        while stop_idx < len(stops_for_route):
            stop_identifier = stops_for_route[stop_idx]
            if stop_identifier not in seen_stops:
                unique_stop_list.append(stop_identifier)
                seen_stops.add(stop_identifier)
            stop_idx += 1
        route_to_stops[route_identifier] = unique_stop_list
        route_counter += 1

    # Count trips per stop
    stop_trip_counts = df_stop_times['stop_id'].value_counts().to_dict()
    stop_keys = list(stop_trip_counts.keys())
    stop_idx = 0
    while stop_idx < len(stop_keys):
        stop_identifier = stop_keys[stop_idx]
        count = stop_trip_counts[stop_identifier]
        stop_trip_count[stop_identifier] = count
        stop_idx += 1


def get_busiest_routes():
    """
    Get the top 5 busiest routes based on the number of trips.
    """
    # Count trips per route using the trip_to_route mapping
    route_trip_counter = defaultdict(int)
    trip_keys = list(trip_to_route.keys())
    idx = 0

    while idx < len(trip_keys):
        trip = trip_keys[idx]
        route = trip_to_route[trip]
        route_trip_counter[route] += 1
        idx += 1

    # Sort the routes by trip count in descending order and get the top 5
    sorted_routes = sorted(route_trip_counter.items(), key=lambda entry: entry[1], reverse=True)
    return sorted_routes[:5]

def get_most_frequent_stops():
    """
    Get the top 5 stops with the most frequent trips.
    """
    # Sort stop_trip_count by the number of trips in descending order
    stop_freq_list = sorted(stop_trip_count.items(), key=lambda element: element[1], reverse=True)
    return stop_freq_list[:5]

def get_top_5_busiest_stops():
    """
    Get the top 5 stops with the highest number of routes passing through them.
    """
    # Map each stop to the set of routes passing through it
    stop_routes_map = defaultdict(set)
    route_ids = list(route_to_stops.keys())
    idx = 0

    while idx < len(route_ids):
        route = route_ids[idx]
        stops = route_to_stops[route]
        stop_idx = 0

        while stop_idx < len(stops):
            stop = stops[stop_idx]
            stop_routes_map[stop].add(route)
            stop_idx += 1

        idx += 1

    # Count the number of routes for each stop
    stop_route_counts = {stop: len(routes) for stop, routes in stop_routes_map.items()}

    # Sort stops by route count in descending order and get the top 5
    sorted_stops = sorted(stop_route_counts.items(), key=lambda item: item[1], reverse=True)
    return sorted_stops[:5]

def get_stops_with_one_direct_route():
    """
    Get the top 5 pairs of stops connected by exactly one direct route, sorted by trip frequency.
    """
    # Map pairs of stops to the routes connecting them
    stop_pair_to_routes = defaultdict(list)
    route_ids = list(route_to_stops.keys())
    idx = 0

    while idx < len(route_ids):
        route = route_ids[idx]
        stops = route_to_stops[route]
        stop_idx = 0

        while stop_idx < len(stops) - 1:
            stop_a = stops[stop_idx]
            stop_b = stops[stop_idx + 1]
            stop_pair_to_routes[(stop_a, stop_b)].append(route)
            stop_idx += 1

        idx += 1

    # Filter pairs with exactly one direct route
    single_route_stop_pairs = {
        pair: routes[0] for pair, routes in stop_pair_to_routes.items() if len(routes) == 1
    }

    # Calculate the combined trip frequency for each stop pair
    combined_trip_frequencies = []
    stop_pairs = list(single_route_stop_pairs.keys())
    idx = 0

    while idx < len(stop_pairs):
        stop_pair = stop_pairs[idx]
        route = single_route_stop_pairs[stop_pair]
        stop_a, stop_b = stop_pair
        combined_freq = stop_trip_count[stop_a] + stop_trip_count[stop_b]
        combined_trip_frequencies.append((stop_pair, route, combined_freq))
        idx += 1

    # Sort by combined frequency in descending order and get the top 5
    sorted_frequencies = sorted(combined_trip_frequencies, key=lambda entry: entry[2], reverse=True)
    return sorted_frequencies[:5]




# Function to get merged fare DataFrame
# No need to change this function
def get_merged_fare_df():
    """
    Retrieve the merged fare DataFrame.

    Returns:
        DataFrame: The merged fare DataFrame containing fare rules and attributes.
    """
    global merged_fare_df
    return merged_fare_df

# Visualize the stop-route graph interactively
# Visualize the stop-route graph interactively
def visualize_stop_route_graph(route_to_stops):
    """
    Visualize the stop-route graph interactively using Plotly.
    """
    import networkx as nx
    import plotly.graph_objects as go

    graph = nx.Graph()

    # Add edges between stops
    route_keys = list(route_to_stops.keys())
    index = 0
    while index < len(route_keys):
        stop_sequence = route_to_stops[route_keys[index]]
        stop_idx = 0
        while stop_idx < len(stop_sequence) - 1:
            graph.add_edge(stop_sequence[stop_idx], stop_sequence[stop_idx + 1])
            stop_idx += 1
        index += 1

    # Compute positions for nodes
    layout_positions = nx.spring_layout(graph, seed=42)

    edge_x_coords, edge_y_coords = [], []
    edge_list = list(graph.edges())
    edge_counter = 0

    while edge_counter < len(edge_list):
        edge = edge_list[edge_counter]
        node1_x, node1_y = layout_positions[edge[0]]
        node2_x, node2_y = layout_positions[edge[1]]
        edge_x_coords.extend([node1_x, node2_x, None])
        edge_y_coords.extend([node1_y, node2_y, None])
        edge_counter += 1

    edge_trace = go.Scatter(
        x=edge_x_coords,
        y=edge_y_coords,
        line=dict(width=0.6, color='#1234CC'),
        hoverinfo='none',
        mode='lines'
    )

    node_x_coords, node_y_coords = [], []
    node_labels = []
    node_list = list(graph.nodes())
    node_counter = 0

    while node_counter < len(node_list):
        node = node_list[node_counter]
        x_coord, y_coord = layout_positions[node]
        node_x_coords.append(x_coord)
        node_y_coords.append(y_coord)
        node_labels.append(f"ID: {node}")
        node_counter += 1

    node_trace = go.Scatter(
        x=node_x_coords,
        y=node_y_coords,
        mode='markers+text',
        text=node_labels,
        textposition="top center",
        hoverinfo='text'
    )

    figure = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title='Visualization of Stops and Routes Graph',
        )
    )

    figure.show()



# Brute-Force Approach for finding direct routes
def direct_route_brute_force(start_stop, end_stop):
    """
    Find all direct routes between the start and end stop using a brute-force approach.
    Tracks the number of steps taken during computation.
    """
    found_routes = []
    route_keys = list(route_to_stops.keys())
    steps = 0  # Step counter
    idx = 0

    while idx < len(route_keys):
        current_route = route_keys[idx]
        steps += 1  # Increment for accessing a route
        stops_list = route_to_stops[current_route]
        
        if start_stop in stops_list and end_stop in stops_list:
            steps += 1  # Increment for checking stop presence
            start_pos = stops_list.index(start_stop)
            end_pos = stops_list.index(end_stop)
            steps += 2  # Increment for indexing operations

            if start_pos < end_pos:
                found_routes.append(current_route)
                steps += 1  # Increment for appending a valid route

        idx += 1
        steps += 1  # Increment for loop iteration

    direct_route_brute_force.steps = steps  # Attach the step count to the function
    return found_routes




# Initialize Datalog predicates for reasoning
# Initialize Datalog predicates for reasoning
pyDatalog.create_terms('RouteHasStop, DirectRoute, OptimalRoute, X, Y, Z, R, R1, R2')

def initialize_datalog():
    pyDatalog.clear()
    add_route_data(route_to_stops)
    pyDatalog.create_terms('RouteHasStop, DirectRoute, X, Y, R')
    DirectRoute(R, X, Y) <= RouteHasStop(R, X) & RouteHasStop(R, Y) & (X != Y)

# Adding route data to Datalog
def add_route_data(route_to_stops):
    """
    Add route and stop data to the Datalog engine.
    Each route is linked to its stops.
    """
    route_keys = list(route_to_stops.keys())  # Get all route IDs
    route_index = 0  # Initialize route index for iteration

    while route_index < len(route_keys):  # Iterate through each route
        current_route = route_keys[route_index]  # Get the current route ID
        stop_list = route_to_stops[current_route]  # Get the stops for the route

        stop_index = 0  # Initialize stop index for iteration
        while stop_index < len(stop_list):  # Iterate through stops in the route
            current_stop = stop_list[stop_index]  # Get the current stop
            +RouteHasStop(current_route, current_stop)  # Add to Datalog
            stop_index += 1  # Move to the next stop

        route_index += 1  # Move to the next route

# Function to query direct routes between two stops
def query_direct_routes(start, end):
    """
    Query all direct routes that connect the start and end stops.
    Tracks the number of steps taken during computation.
    """
    # Query the DirectRoute relationship for routes between the two stops
    direct_routes = DirectRoute(R, start, end)
    steps = 0  # Initialize step counter
    routes_index = 0  # Initialize index for extracting results
    steps += 1  # Increment for initializing query and index

    # Initialize an empty list to store the matching route IDs
    result_routes = []
    steps += 1  # Increment for initializing result_routes

    # Use a while loop to process each route in the query result
    while routes_index < len(direct_routes):
        steps += 1  # Increment for loop condition check
        current_route = direct_routes[routes_index][0]  # Extract route ID
        steps += 1  # Increment for extracting route ID
        result_routes.append(current_route)  # Add to the results list
        steps += 1  # Increment for appending to result_routes
        routes_index += 1  # Move to the next result
        steps += 1  # Increment for index increment

    query_direct_routes.steps = steps  # Attach the step count to the function
    return result_routes  # Return the list of direct route IDs





# Forward chaining for optimal route planning
def forward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Find routes using forward chaining, with step tracking.

    Args:
        start_stop_id (int): Starting stop ID.
        end_stop_id (int): Ending stop ID.
        stop_id_to_include (int): Stop ID that must be included in the route.
        max_transfers (int): Maximum number of allowed transfers.

    Returns:
        list: Sorted list of tuples [(route1, stop_id_to_include, route2), ...].
    """
    steps = 0  # Step counter

    # Clear previous Datalog facts and rules
    pyDatalog.clear()
    steps += 1  # Increment for clearing Datalog facts

    add_route_data(route_to_stops)  # Add route-stop data to Datalog
    steps += 1  # Increment for adding route data

    # Define Datalog terms for rules and queries
    pyDatalog.create_terms('RouteHasStop, DirectRoute, R1, R2')
    steps += 1  # Increment for defining Datalog terms

    # Define the DirectRoute relationship
    DirectRoute(R1, stop_id_to_include, R2) <= (
        RouteHasStop(R1, start_stop_id) &
        RouteHasStop(R1, stop_id_to_include) &
        RouteHasStop(R2, end_stop_id) &
        RouteHasStop(R2, stop_id_to_include) &
        (R1 != R2)
    )
    steps += 1  # Increment for defining the DirectRoute relationship

    # Query the DirectRoute relationship
    query_results = DirectRoute(R1, stop_id_to_include, R2)
    steps += 1  # Increment for executing the query

    index = 0  # Initialize index for iteration
    steps += 1  # Increment for index initialization

    # Prepare results with a while loop
    extracted_results = []
    steps += 1  # Increment for initializing the results list

    while index < len(query_results):  # Iterate over query results
        steps += 1  # Increment for loop condition check
        current_row = query_results[index]  # Get the current result row
        steps += 1  # Increment for accessing query result
        current_r1, current_r2 = current_row[0], current_row[1]  # Extract route IDs
        steps += 1  # Increment for extracting route IDs

        # Apply transfer constraint
        if max_transfers >= 1:
            extracted_results.append((current_r1, stop_id_to_include, current_r2))
            steps += 1  # Increment for appending to results

        index += 1  # Move to the next result
        steps += 1  # Increment for index increment

    steps += 1  # Increment for sorting the results
    forward_chaining.steps = steps  # Attach the step count to the function
    return sorted(extracted_results)  # Return the sorted results as a list of tuples




# Backward chaining for optimal route planning
def backward_chaining(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Finds possible routes using backward chaining logic with a specific intermediate stop.
    Returns results as tuples to match the expected format and tracks the steps taken.
    """
    steps = 0  # Initialize step counter

    # Clear previous Datalog facts and rules
    pyDatalog.clear()
    steps += 1  # Increment for clearing Datalog facts

    add_route_data(route_to_stops)  # Add route-stop data to Datalog
    steps += 1  # Increment for adding route data

    # Define Datalog terms for rules and queries
    pyDatalog.create_terms('RouteHasStop, DirectRoute, R1, R2')
    steps += 1  # Increment for defining Datalog terms

    # Define the DirectRoute relationship
    DirectRoute(R1, stop_id_to_include, R2) <= (
        RouteHasStop(R1, start_stop_id) &
        RouteHasStop(R1, stop_id_to_include) &
        RouteHasStop(R2, end_stop_id) &
        RouteHasStop(R2, stop_id_to_include) &
        (R1 != R2)
    )
    steps += 1  # Increment for defining the DirectRoute relationship

    # Query the DirectRoute relationship
    querydic = DirectRoute(R1, stop_id_to_include, R2)
    steps += 1  # Increment for executing the query

    # Adjust output to match expected format: reverse and convert to tuples
    result = []
    steps += 1  # Increment for initializing the results list

    for row in querydic:  # Loop through query results
        steps += 1  # Increment for loop iteration
        if max_transfers >= 1:  # Check the transfer constraint
            result.append((int(row[1]), int(stop_id_to_include), int(row[0])))
            steps += 1  # Increment for appending to results

    steps += 1  # Increment for returning the result
    backward_chaining.steps = steps  # Attach the step count to the function
    return result  # Return the results


# PDDL-style planning for route finding
def pddl_planning(start_stop_id, end_stop_id, stop_id_to_include, max_transfers):
    """
    Implements PDDL-like forward planning for routes using PyDatalog, with step tracking.

    Args:
        start_stop_id (int): Starting stop ID.
        end_stop_id (int): Ending stop ID.
        stop_id_to_include (int): Stop ID to include in the route.
        max_transfers (int): Maximum number of allowed transfers.

    Returns:
        tuple: A tuple containing:
            - list: A list of tuples (route1, stop_id_to_include, route2) representing valid routes.
            - int: Number of steps taken during the process.
    """
    pyDatalog.clear()
    steps = 0  # Initialize the step counter
    steps += 1  # Step for clearing previous Datalog data

    add_route_data(route_to_stops)  # Add route-stop data to Datalog
    steps += 1  # Step for adding route-stop data

    pyDatalog.create_terms('RouteHasStop, DirectRoute, R1, R2')
    steps += 1  # Step for defining terms and rules

    # Define the DirectRoute relationship
    DirectRoute(R1, stop_id_to_include, R2) <= (
        RouteHasStop(R1, start_stop_id) &
        RouteHasStop(R1, stop_id_to_include) &
        RouteHasStop(R2, end_stop_id) &
        RouteHasStop(R2, stop_id_to_include) &
        (R1 != R2)
    )
    steps += 1  # Step for defining the DirectRoute rule

    # Query the routes
    ans = DirectRoute(R1, stop_id_to_include, R2)
    steps += 1  # Step for querying the routes

    # Adjust output to match the expected format
    res = [(int(row[0]), int(stop_id_to_include), int(row[1])) for row in ans if max_transfers >= 1]
    steps += len(ans)  # Count processing each result row as a step

    # Return results along with the step count
    return res if res else [], steps



# Function to filter fare data based on an initial fare limit
def prune_data(merged_fare_df, initial_fare):
    """
    Filter fare data based on an initial fare limit.

    Args:
        merged_fare_df (DataFrame): The merged fare DataFrame.
        initial_fare (float): The maximum fare allowed.

    Returns:
        DataFrame: A filtered DataFrame containing only routes within the fare limit.
    """
    pass  # Implementation here

# Pre-computation of Route Summary
def compute_route_summary(pruned_df):
    """
    Generate a summary of routes based on fare information.

    Args:
        pruned_df (DataFrame): The filtered DataFrame containing fare information.

    Returns:
        dict: A summary of routes with the following structure:
              {
                  route_id (int): {
                      'min_price': float,          # The minimum fare for the route
                      'stops': set                # A set of stop IDs for that route
                  }
              }
    """
    pass  # Implementation here

# BFS for optimized route planning
def bfs_route_planner_optimized(start_stop_id, end_stop_id, initial_fare, route_summary, max_transfers=3):
    """
    Use Breadth-First Search (BFS) to find the optimal route while considering fare constraints.

    Args:
        start_stop_id (int): The starting stop ID.
        end_stop_id (int): The ending stop ID.
        initial_fare (float): The available fare for the trip.
        route_summary (dict): A summary of routes with fare and stop information.
        max_transfers (int): The maximum number of transfers allowed (default is 3).

    Returns:
        list: A list representing the optimal route with stops and routes taken, structured as:
              [
                  (route_id (int), stop_id (int)),  # Tuple for each stop taken in the route
                  ...
              ]
    """
    pass  # Implementation here


def main():
    # Create the knowledge base
    create_kb()

    # Answer the questions
    print("Top 5 busiest routes:", get_busiest_routes())
    print("Top 5 most frequent stops:", get_most_frequent_stops())
    print("Top 5 busiest stops:", get_top_5_busiest_stops())
    print("Top 5 stop pairs with one direct route:", get_stops_with_one_direct_route())

    # Plot the graph
    visualize_stop_route_graph(route_to_stops)

if __name__ == "__main__":
    main()
