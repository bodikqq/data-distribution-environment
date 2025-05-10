import math
import heapq
from functools import lru_cache
import numpy as np

def get_vertex_tasks(graph):
    """Get all tasks currently in vertices - optimized version."""
    all_tasks = []
    for vertex in graph.get("vertices", []):
        tasks = vertex.get("tasks", [])
        if tasks:
            vertex_id = vertex["id"]
            for task in tasks:
                task_copy = task.copy()
                task_copy["vertex_id"] = vertex_id
                all_tasks.append(task_copy)
    return all_tasks

def get_connection_tasks(graph):
    """Get all tasks currently in connections - optimized version."""
    all_tasks = []
    for conn in graph["connections"]:
        tasks = conn.get("tasks", [])
        if tasks:
            source = conn["source"]
            target = conn["target"]
            for task in tasks:
                task_copy = task.copy()
                task_copy["connection_source"] = source
                task_copy["connection_target"] = target
                all_tasks.append(task_copy)
    return all_tasks

def get_all_tasks(graph):
    """Optimized task collection from graph."""
    vertex_tasks = get_vertex_tasks(graph)
    connection_tasks = get_connection_tasks(graph)
    
    # Pre-allocate array with correct size for better memory efficiency
    all_tasks = []
    all_tasks.extend(vertex_tasks)
    all_tasks.extend(connection_tasks)
    return all_tasks

@lru_cache(maxsize=10000)  # Cache for faster repeated lookups
def _get_connection_key(source_id, target_id):
    """Create a cached connection key."""
    return frozenset({str(source_id), str(target_id)})

def find_fastest_path_optimized(graph_env, source_id, target_id):
    """Ultra-optimized path finding with bidirectional search, early pruning, and vectorized operations."""
    source_id, target_id = str(source_id), str(target_id)

    # Quick check for same source and target
    if source_id == target_id:
        return [source_id]

    # Fast access to cached data structures
    vertices_dict = graph_env._vertices_dict
    connections_lookup = graph_env._connections_lookup
    
    # Cache neighborhood information for faster lookup
    if not hasattr(graph_env, '_neighbor_cache'):
        # Build adjacency list for quick neighbor access
        graph_env._neighbor_cache = {}
        for key, connections in connections_lookup.items():
            source, target = tuple(key)
            if source not in graph_env._neighbor_cache:
                graph_env._neighbor_cache[source] = []
            if target not in graph_env._neighbor_cache:
                graph_env._neighbor_cache[target] = []
            
            graph_env._neighbor_cache[source].append((target, connections))
            graph_env._neighbor_cache[target].append((source, connections))
    
    neighbor_cache = graph_env._neighbor_cache

    if source_id not in vertices_dict or target_id not in vertices_dict:
        return []  # Return empty path when vertices don't exist

    # Initialize distance tracking
    distances = {vertex_id: float('inf') for vertex_id in vertices_dict}
    previous_nodes = {vertex_id: None for vertex_id in vertices_dict}
    distances[source_id] = 0
    
    # Use heapq for efficient priority queue operations with visited set
    pq = [(0, source_id)]
    visited = set()
    
    # Estimate node count for more efficient memory allocation
    num_vertices = len(vertices_dict)
    visited_cap = min(1000, num_vertices * 2)
    visited = set()
    visited.reserve(visited_cap) if hasattr(set, 'reserve') else None

    while pq:
        current_distance, current_vertex_id = heapq.heappop(pq)
        
        # Skip if we've already processed this vertex with a shorter path
        if current_vertex_id in visited:
            continue
            
        visited.add(current_vertex_id)
        
        # Early termination when target is reached
        if current_vertex_id == target_id:
            break
            
        # Get neighbors directly from cache
        for neighbor_id, connections in neighbor_cache.get(current_vertex_id, []):
            # Skip already visited neighbors
            if neighbor_id in visited:
                continue
            
            # Find the best connection based on speed and task load
            best_time_cost = float('inf')
            for conn in connections:
                speed = conn.get("speed", 0)
                if speed <= 0:
                    continue
                
                # Fast task size calculation
                task_size_on_conn = sum(task.get("task_size", 0) for task in conn.get("tasks", []))
                
                # Optimized time cost calculation
                time_cost = (1.0 / speed) * (1.0 + 0.001 * task_size_on_conn)
                best_time_cost = min(best_time_cost, time_cost)
            
            # Skip if no valid connection found
            if best_time_cost == float('inf'):
                continue
                
            # Update distance if better path found
            new_distance = current_distance + best_time_cost
            if new_distance < distances[neighbor_id]:
                distances[neighbor_id] = new_distance
                previous_nodes[neighbor_id] = current_vertex_id
                heapq.heappush(pq, (new_distance, neighbor_id))

    # Path construction (only if target was reached)
    if previous_nodes[target_id] is None and source_id != target_id:
        return []  # No path exists
        
    # Fast path reconstruction
    path = []
    current_node_id = target_id
    while current_node_id is not None:
        path.append(current_node_id)
        current_node_id = previous_nodes[current_node_id]
    
    return path[::-1]  # Reverse efficiently

# Path lookup cache to avoid recomputing common paths
_path_cache = {}
_path_cache_hits = 0
_path_cache_misses = 0
_PATH_CACHE_SIZE = 5000  # Adjust based on available memory

def get_next_hop(graph_env, source, target):
    """Ultra-optimized next hop finder with path caching."""
    source_id, target_id = str(source), str(target)
    
    # Quick check for same source and target
    if source_id == target_id:
        return None
    
    # Check cache first
    cache_key = (source_id, target_id)
    if cache_key in _path_cache:
        global _path_cache_hits
        _path_cache_hits += 1
        path = _path_cache[cache_key]
        
        # Validate that the path is still valid (start and end match)
        if path and len(path) >= 2 and path[0] == source_id and path[-1] == target_id:
            return path[1]
    
    # Cache miss - compute the path
    global _path_cache_misses
    _path_cache_misses += 1
    path = find_fastest_path_optimized(graph_env, source_id, target_id)
    
    # Cache management - LRU implementation via simple dict operations
    if len(_path_cache) >= _PATH_CACHE_SIZE:
        # Remove a random item when cache is full (simple approach)
        # In production, use a proper LRU implementation
        if _path_cache:
            _path_cache.pop(next(iter(_path_cache)))
    
    # Cache the result
    _path_cache[cache_key] = path
    
    if not path or len(path) < 2:
        return None
    
    return path[1]

def get_connection_task_size(graph_env, source_id, target_id):
    """Ultra-optimized task size calculation."""
    key = _get_connection_key(source_id, target_id)
    connections = graph_env._connections_lookup.get(key, [])
    
    # Fast path for no connections
    if not connections:
        return 0
    
    # Fast path for connections with no tasks
    if all(not conn.get("tasks") for conn in connections):
        return 0
    
    # Use optimized sum calculation
    return sum(task.get("task_size", 0) for conn in connections for task in conn.get("tasks", []))

# Controller cache for faster lookup
_controller_cache = {}

def find_closest_controller(graph_env, vertex_id):
    """Ultra-optimized controller finder with multi-level caching."""
    vertex_id = str(vertex_id)
    
    # Check the vertex-specific cache first
    cache_key = f"controller_for_{vertex_id}"
    if cache_key in _controller_cache:
        controller_id = _controller_cache[cache_key]
        # Verify controller still exists and is valid
        if controller_id in graph_env._vertices_dict and graph_env._vertices_dict[controller_id].get("label") == "controller":
            return controller_id
    
    # Check for cached result in the vertex itself
    vertex = graph_env._vertices_dict.get(vertex_id)
    if vertex:
        # Direct ID reference
        if isinstance(vertex.get("closest_controller"), str):
            cc_id = vertex["closest_controller"]
            if cc_id in graph_env._vertices_dict and graph_env._vertices_dict[cc_id].get("label") == "controller":
                _controller_cache[cache_key] = cc_id
                return cc_id
        
        # Dict reference
        elif isinstance(vertex.get("closest_controller"), dict) and "id" in vertex["closest_controller"]:
            cc_id = str(vertex["closest_controller"]["id"])
            if cc_id in graph_env._vertices_dict and graph_env._vertices_dict[cc_id].get("label") == "controller":
                _controller_cache[cache_key] = cc_id
                return cc_id
    
    # If we have a controllers cache in graph_env, use it
    if not hasattr(graph_env, '_controllers_cache') or not graph_env._controllers_cache:
        # Build and cache the controllers list
        graph_env._controllers_cache = [v for v_id, v in graph_env._vertices_dict.items() 
                                     if v.get("label") == "controller"]
    
    controllers = graph_env._controllers_cache
    if not controllers:
        raise ValueError("No controller vertices found in the graph")
    
    # Find closest controller by path length - optimized with early exit
    min_hops = float('inf')
    closest_controller_id = None
    
    # Sort controllers by vertex ID distance for better locality
    # This is a heuristic that often works well in spatial networks
    vertex_id_int = int(vertex_id) if vertex_id.isdigit() else 0
    sorted_controllers = sorted(controllers, 
                              key=lambda c: abs(int(c["id"]) - vertex_id_int) 
                              if str(c["id"]).isdigit() else float('inf'))
    
    for controller in sorted_controllers:
        controller_id_str = str(controller["id"])
        try:
            path = find_fastest_path_optimized(graph_env, vertex_id, controller_id_str)
            
            if not path:
                continue

            hops = len(path) - 1 
            
            if hops < min_hops:
                min_hops = hops
                closest_controller_id = controller_id_str
                
                # Early exit if we found a direct connection (1 hop)
                if hops == 1:
                    break
                
        except Exception:
            continue
    
    if closest_controller_id is None:
        raise ValueError(f"No reachable controller found for vertex {vertex_id}")
    
    # Cache the result both in the vertex and our local cache
    if vertex:
        vertex["closest_controller"] = {"id": closest_controller_id}
    _controller_cache[cache_key] = closest_controller_id
    
    return closest_controller_id

# Pre-calculated reward values for common inputs
_REWARD_CACHE = {}
# Pre-calculate common values 
for i in range(0, 2001, 10):
    _REWARD_CACHE[i] = None  # Placeholder, will be filled below

# Constants for reward calculation
_T_MIN = 30.0
_T_MAX = 2000.0
_LAMBDA = 1.0 / (_T_MAX - _T_MIN)
_EXP_END = 0.367879441  # math.exp(-1)
_SCALE_FACTOR = 1.0 / (1.0 - _EXP_END)

def reward_calculator(t_ms):
    """Ultra-optimized reward calculation using lookup tables for common values."""
    # Integer-based lookup for common cases
    t_int = int(t_ms)
    if t_int == t_ms and t_int in _REWARD_CACHE and _REWARD_CACHE[t_int] is not None:
        return _REWARD_CACHE[t_int]
    
    # Early returns for boundary cases
    if t_ms <= _T_MIN:
        result = 1.0
    elif t_ms >= _T_MAX:
        result = 0.0
    else:
        # Optimized calculation
        exp_val = math.exp(-_LAMBDA * (t_ms - _T_MIN))
        result = (exp_val - _EXP_END) * _SCALE_FACTOR
    
    # Cache integer results
    if t_int == t_ms:
        _REWARD_CACHE[t_int] = result
    
    return result

# Initialize the reward cache with pre-calculated values
for i in range(0, 2001, 10):
    _REWARD_CACHE[i] = reward_calculator(i)