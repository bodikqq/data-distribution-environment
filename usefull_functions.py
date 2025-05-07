import math
def get_vertex_tasks(graph):
    """Get all tasks currently in vertices."""
    all_tasks = []
    for vertex in graph.get("vertices", []):
        for task in vertex.get("tasks", []):
            task_copy = task.copy()
            task_copy["vertex_id"] = vertex["id"]
            all_tasks.append(task_copy)
    return all_tasks

def get_connection_tasks(graph):
    """Get all tasks currently in connections."""
    all_tasks = []
    for conn in graph["connections"]:
        for task in conn.get("tasks", []):
            task_copy = task.copy()
            task_copy["connection_source"] = conn["source"]
            task_copy["connection_target"] = conn["target"]
            all_tasks.append(task_copy)
    return all_tasks

def get_all_tasks(graph):
    return get_vertex_tasks(graph) + get_connection_tasks(graph)

def find_fastest_path_optimized(graph_env, source_id, target_id):
    """Optimized path finding using pre-calculated lookups from GraphEnv."""
    source_id, target_id = str(source_id), str(target_id)

    # Access pre-built structures from graph_env
    vertices_dict = graph_env._vertices_dict
    connections_lookup = graph_env._connections_lookup

    if source_id not in vertices_dict or target_id not in vertices_dict:
        raise ValueError(f"Source or target ID not in graph: {source_id} -> {target_id}")

    distances = {vertex_id: float('inf') for vertex_id in vertices_dict}
    previous_nodes = {vertex_id: None for vertex_id in vertices_dict}
    distances[source_id] = 0
    
    # Priority queue: (distance, vertex_id)
    pq = [(0, source_id)]

    while pq:
        current_distance, current_vertex_id = min(pq, key=lambda x: x[0]) # Simplified min-heap behavior
        pq = [item for item in pq if item[1] != current_vertex_id] # Remove current_vertex_id


        if current_vertex_id == target_id:
            break # Target reached

        # Iterate over connections involving current_vertex_id
        # We need to check all connections and see if current_vertex_id is a source or target
        
        # Collect neighbors and their connection properties
        neighbors_to_process = []
        for key, conn_list in connections_lookup.items():
            if current_vertex_id in key: # current_vertex_id is part of this connection key
                for conn in conn_list: # Iterate through all connections for this pair
                    # Determine the neighbor
                    neighbor_id = None
                    if str(conn["source"]) == current_vertex_id:
                        neighbor_id = str(conn["target"])
                    elif str(conn["target"]) == current_vertex_id:
                        neighbor_id = str(conn["source"])
                    
                    if neighbor_id and neighbor_id in distances: # Ensure neighbor is part of the graph
                        neighbors_to_process.append({
                            "id": neighbor_id,
                            "speed": conn["speed"],
                            "tasks": conn.get("tasks", []) # Get tasks for this specific connection
                        })
        
        for neighbor_info in neighbors_to_process:
            neighbor_id = neighbor_info["id"]
            
            # Calculate task size on this specific connection
            task_size_on_conn = 0
            for task in neighbor_info["tasks"]:
                task_size_on_conn += task.get("task_size", 0)

            # Calculate time (cost) for this connection
            # Ensure speed is not zero to avoid division by zero
            if neighbor_info["speed"] <= 0:
                time_cost = float('inf')
            else:
                time_cost = (1 / neighbor_info["speed"]) * (1 + task_size_on_conn / 1000.0)
            
            new_distance = current_distance + time_cost

            if new_distance < distances[neighbor_id]:
                distances[neighbor_id] = new_distance
                previous_nodes[neighbor_id] = current_vertex_id
                # Add to pq or update if already present (simplified: just add)
                pq.append((new_distance, neighbor_id))

    # Reconstruct path
    path = []
    current_node_id = target_id
    while current_node_id is not None:
        path.append(current_node_id)
        current_node_id = previous_nodes[current_node_id]
    
    if not path or path[-1] != source_id: # Check if a path was found
        # Instead of raising an error, return an empty list or specific indicator
        # This allows the caller (e.g., get_next_hop) to handle "no path" gracefully
        return [] 
        # raise ValueError(f"No path exists from {source_id} to {target_id}")

    return list(reversed(path))


def get_next_hop(graph_env, source, target):
    """Get the next hop in the fastest path from source to target using the optimized pathfinder."""
    # Path is now found using the optimized function that takes graph_env
    path = find_fastest_path_optimized(graph_env, source, target) 
    
    if len(path) < 2:
        # This can mean target is source, or no path exists.
        # The find_fastest_path_optimized now returns [] for no path.
        # Consider how to differentiate or if it matters for the caller.
        # For now, raising an error or returning None/special value might be appropriate.
        # print(f'Warning: Target is the source or no path exists. Source: {source}, Target: {target}, Path: {path}')
        return None # Or raise an error if that's preferred behavior
    
    return path[1]  # The next hop is the second vertex in the path

def get_connection_task_size(graph_env, source_id, target_id):
    """Get total task size on connections between source_id and target_id."""
    source_id, target_id = str(source_id), str(target_id)
    total_weight = 0
    
    # Use the pre-built lookup from graph_env
    connections_lookup = graph_env._connections_lookup
    key = frozenset({source_id, target_id})
    
    if key in connections_lookup:
        for connection in connections_lookup[key]: # Iterate through all connections for this pair
            for task in connection.get("tasks", []):
                total_weight += task.get("task_size", 0)
    return total_weight


def find_closest_controller(graph_env, vertex_id):
    """Find the closest controller vertex to the given vertex based on number of hops,
       using the optimized pathfinder."""
    vertex_id = str(vertex_id)
    
    # Access pre-built structures from graph_env
    vertices_dict = graph_env._vertices_dict
    
    # First check if the vertex has the pre-calculated closest_controller (if that feature is kept)
    vertex_data = vertices_dict.get(vertex_id)
    if vertex_data and "closest_controller" in vertex_data and isinstance(vertex_data["closest_controller"], dict) and "id" in vertex_data["closest_controller"]:
        # Ensure it's a valid reference before returning
        cc_id = str(vertex_data["closest_controller"]["id"])
        if cc_id in vertices_dict and vertices_dict[cc_id].get("label") == "controller":
            return cc_id

    min_hops = float('inf')
    closest_controller_id = None
    
    controllers = [v for v_id, v in vertices_dict.items() if v.get("label") == "controller"]
    
    if not controllers:
        raise ValueError("No controller vertices found in the graph")
    
    for controller in controllers:
        controller_id_str = str(controller["id"])
        try:
            # Use the optimized pathfinder that takes graph_env
            path = find_fastest_path_optimized(graph_env, vertex_id, controller_id_str)
            
            if not path: # No path found
                continue

            hops = len(path) - 1 
            
            if hops < min_hops:
                min_hops = hops
                closest_controller_id = controller_id_str
                
        except ValueError: # Catch errors from pathfinding (e.g., vertex not in graph)
            continue 
    
    if closest_controller_id is None:
        raise ValueError(f"No reachable controller found for vertex {vertex_id}")
        
    return closest_controller_id

def reward_calculator(t_ms: float) -> float:
    t_min = 30.0      # below this → full reward
    t_max = 2000.0    # at or above this → zero reward

    if t_ms <= t_min:
        return 1.0
    if t_ms >= t_max:
        return 0.0

    # exponential decay factor over the window [t_min, t_max]
    # here we choose λ so that
    # exp(-λ*(t_max - t_min)) = e^{-1} ≈ 0.367; 
    # you can increase λ for a steeper drop (e.g. use 2/(t_max-t_min) for exp(-2) at the end)
    lambda_param = 1.0 / (t_max - t_min)

    # raw exponential value (ranges between 1 and exp(-1))
    exp_val = math.exp(-lambda_param * (t_ms - t_min))
    exp_end = math.exp(-lambda_param * (t_max - t_min))  # == exp(-1)

    # normalize so that at t_min → 1, at t_max → 0
    return (exp_val - exp_end) / (1.0 - exp_end)