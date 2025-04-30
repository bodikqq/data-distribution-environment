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

def find_fastest_path(graph, source, target):
    """Optimized path finding with early termination."""
    source, target = str(source), str(target)
    
    distances = {}
    previous = {}
    unvisited = set()
    
    # Initialize only necessary vertices
    for conn in graph["connections"]:
        src, tgt = str(conn["source"]), str(conn["target"])
        if src not in distances:
            distances[src] = float('inf')
            previous[src] = None
            unvisited.add(src)
        if tgt not in distances:
            distances[tgt] = float('inf')
            previous[tgt] = None
            unvisited.add(tgt)
    
    distances[source] = 0
    
    while unvisited and target in unvisited:
        current = min((v for v in unvisited), key=lambda x: distances[x])
        
        if current == target:
            break
            
        unvisited.remove(current)
        
        for conn in graph["connections"]:
            src, tgt = str(conn["source"]), str(conn["target"])
            
            if current == src:
                neighbor = tgt
            elif current == tgt:
                neighbor = src
            else:
                continue
                
            if neighbor in unvisited:
                task_size = get_connection_task_size(graph, current, neighbor)
                time = (1 / conn["speed"]) * (1 + task_size/1000.0)
                new_dist = distances[current] + time
                
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
    
    if target not in previous and target != source:
        raise ValueError(f"No path exists from {source} to {target}")
        
    path = []
    current = target
    while current is not None:
        path.append(current)
        current = previous[current]
    
    return list(reversed(path))

def get_next_hop(graph, source, target):
    """Get the next hop in the fastest path from source to target."""
    path = find_fastest_path(graph, source, target)
    
    if len(path) < 2:
        raise ValueError(f'target is the source or no path exists \n source: {source}, target: {target} path: {path}')
    
    return path[1]  # The next hop is the second vertex in the path

def get_connection_task_size(graph, source_id, target_id):
    source_id, target_id = str(source_id), str(target_id)
    total_weight = 0
    
    for connection in graph["connections"]:
        if ((str(connection["source"]) == source_id and str(connection["target"]) == target_id) or
            (str(connection["source"]) == target_id and str(connection["target"]) == source_id)):
            for task in connection.get("tasks", []):
                total_weight += task.get("task_size", 0)
            break
    return total_weight

def find_closest_controller(graph, vertex_id):
    """Find the closest controller vertex to the given vertex based on number of hops.
    First checks if the vertex has a pre-calculated closest_controller parameter.
    
    Args:
        graph: Graph dictionary containing vertices and connections
        vertex_id: ID of the vertex to find closest controller for
        
    Returns:
        str: ID of the closest controller vertex
        
    Raises:
        ValueError: If no controller is found in the graph
    """
    vertex_id = str(vertex_id)
    
    # First check if the vertex has the pre-calculated closest_controller
    for vertex in graph["vertices"]:
        if str(vertex["id"]) == vertex_id and "closest_controller" in vertex:
            return vertex["closest_controller"]["id"]
    
    # If not pre-calculated, fall back to the original calculation
    min_hops = float('inf')
    closest_controller = None
    
    # Find all controller vertices
    controllers = [vertex for vertex in graph["vertices"] if vertex.get("label") == "controller"]
    
    if not controllers:
        raise ValueError("No controller vertices found in the graph")
    
    # Check each controller to find the closest one
    for controller in controllers:
        try:
            path = find_fastest_path(graph, vertex_id, str(controller["id"]))
            # Number of hops is the number of vertices in the path minus 1
            hops = len(path) - 1
            
            if hops < min_hops:
                min_hops = hops
                closest_controller = str(controller["id"])
                
        except ValueError:
            # Skip if no path exists to this controller
            continue
    
    if closest_controller is None:
        raise ValueError(f"No reachable controller found for vertex {vertex_id}")
        
    return closest_controller
