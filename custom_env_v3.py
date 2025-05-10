import gymnasium as gym
import usefull_functions as usfl_func
import usefull_arrays as usfl_arr
from gymnasium import spaces
from typing import Dict, List, Optional, Set, Union
import json 
import numpy as np
import copy  # Add this import for deepcopy
# data share time is bytes per milisecond
# one time step is 10ms
#stable baseline 3
# kedy vidia že nezasvetilo hneď
class GraphEnv(gym.Env):
    """
    A custom Gym environment that uses a JSON graph as observation.
    
    The observation is a dictionary with:
      - "graph": the full graph loaded from a JSON file
    """
    metadata = {"render.modes": ["human"]}
    

    def __init__(self, descriptions_for_regular_tasks, json_path="graph_output.json", time_step=10, scenario="light"):
        super().__init__() # Ensure superclass is initialized
        self.json_path = json_path
        
        # Load graph data only once during initialization
        with open(self.json_path, 'r') as f:
            self.initial_graph_data = json.load(f)

        # Create a deep copy for the working graph
        self.graph = copy.deepcopy(self.initial_graph_data)

        # Initialize/ensure dynamic properties and build necessary lookups
        self._initialize_graph_elements()

        # Cache frequently accessed data
        self._build_caches()

        self.time_step_in_ms = time_step
        self.scenario = scenario
        usfl_arr.descriptions_for_regular_tasks = descriptions_for_regular_tasks
        
        self.time = 0
        self.task_id = 1  # Initialize task_id
        
        # Initialize tracking variables
        self.reward = 0
        self.tasks_awaiting_confirmation = {}
        self.confirmation_tasks = {}
        self.local_confirmation_counter = 0
        self.non_local_confirmation_counter = 0
        
        self.observation = {
            "graph": self.graph,
        }
        ###print("Environment variables initialized")
        sensor_choices      = 317        # 0-316  (IDs ≥5 valid, 1-4 will be filtered out)
        controller_choices  = 5
        # Define action and observation spaces
        self.action_space = spaces.MultiDiscrete(
            [sensor_choices if i % 2 == 0 else controller_choices for i in range(600)]
)
        self.observation_space = spaces.Box(low=0.0, high=0.0, shape=(1,), dtype=np.float32)
        self.constant_obs = np.array([0.0], dtype=np.float32)   
        # Simplified observation space
        #self.observation_space = spaces.Dict({
        #    "graph": spaces.Dict({
        #        "vertices": spaces.Sequence(
        #            spaces.Dict({
        #                "id": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32), 
        #            })
        #        ),
        #        "connections": spaces.Sequence(
        #            spaces.Dict({
        #                "source": spaces.Discrete(1000),
        #                "target": spaces.Discrete(1000),
        #                "speed": spaces.Box(low=0, high=1000000, shape=(), dtype=np.float32)
        #            })
        #        )
        #    })
        #})
    
    def _initialize_graph_elements(self):
        """Helper to initialize graph elements and build lookups."""
        # Pre-allocate dictionaries
        self._vertices_dict = {}
        self._connections_lookup = {}
        self._controller_vertices = []
        
        # Process all vertices first
        for vertex in self.graph.get("vertices", []):
            vertex_id = str(vertex["id"])
            vertex.setdefault("tasks", [])
            
            # Initialize usedSRAM from initial_graph_data or to 0
            initial_v_data = next((v_init for v_init in self.initial_graph_data.get("vertices", []) 
                                 if str(v_init["id"]) == vertex_id), None)
            
            if initial_v_data:
                vertex["usedSRAM"] = initial_v_data.get("usedSRAM", 0)
            else:
                vertex["usedSRAM"] = 0 # Default

            # Setup controller-specific fields
            if vertex.get("label") == "controller":
                vertex.setdefault("sensors_info", {})
                self._controller_vertices.append(vertex)

            # Add to vertices dictionary for O(1) lookups
            self._vertices_dict[vertex_id] = vertex

        # Process all connections
        for conn in self.graph.get("connections", []):
            conn.setdefault("tasks", [])
            conn.setdefault("reserved_from_previous_step", 0)

            # Create bidirectional connection lookup
            source_id, target_id = str(conn["source"]), str(conn["target"])
            key = frozenset({source_id, target_id})
            
            if key not in self._connections_lookup:
                self._connections_lookup[key] = []
            self._connections_lookup[key].append(conn)

    def _build_caches(self):
        """Build additional caches to speed up frequent operations."""
        # Cache controllers list for faster lookup
        self._controllers_cache = self._controller_vertices
        
        # Cache room data if using room-based scenarios
        self._room_sensors_cache = {}
        if hasattr(usfl_arr, 'rooms'):
            for room_name, room_data in usfl_arr.rooms.items():
                self._room_sensors_cache[room_name] = room_data

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state.
        
        Args:
            seed: Optional seed for random number generator
            options: Optional configuration arguments
            
        Returns:
            observation: Initial observation
            info: Additional information
        """
        super().reset(seed=seed)
        
        # Reset environment counters
        self.time = 0
        self.task_id = 1
        self.reward = 0
        usfl_arr.reset_regular_tasks() # Assuming this is external and correct
        
        # Reset task-related dictionaries
        self.tasks_awaiting_confirmation = {}
        self.confirmation_tasks = {}
        self.local_confirmation_counter = 0
        self.non_local_confirmation_counter = 0
        
        # Reset graph state IN-PLACE
        for vertex_id_str, vertex in self._vertices_dict.items():
            vertex["tasks"] = []
            
            # Restore usedSRAM from the initial template data
            initial_v_data = next((v_init for v_init in self.initial_graph_data.get("vertices", []) if str(v_init["id"]) == vertex_id_str), None)
            if initial_v_data:
                vertex["usedSRAM"] = initial_v_data.get("usedSRAM", 0)
            else:
                # This case should ideally not be hit if _vertices_dict is derived from initial_graph_data
                vertex["usedSRAM"] = 0 

            if vertex.get("label") == "controller":
                vertex["sensors_info"] = {} # Clear sensor info

        # Reset connections tasks and reserved bandwidth
        for conn_list in self._connections_lookup.values():
            for conn in conn_list:
                conn["tasks"] = []
                conn["reserved_from_previous_step"] = 0
        
        # If any part of the graph structure itself (not just dynamic fields) was meant to be reset
        # from graph_copy and _initialize_graph_elements doesn't cover it,
        # self.graph = copy.deepcopy(self.initial_graph_data) would be needed,
        # followed by self._initialize_graph_elements() or just rebuilding lookups.
        # For now, assuming only dynamic fields need reset.

        # Update observation
        #self.observation = {"graph": self.graph}
        
        return self.constant_obs, {} # Return observation and an empty info dict

    def step(self, action):
        """Process an Nx2 matrix where each row contains a sensor ID and target, and advance the environment until all non-info tasks are complete.
        
        Args:
            matrix: An Nx2 numpy array or list of lists where each row contains:
                   - Column 0: sensor ID
                   - Column 1: target ID
                   

        Returns:
            observation: Current state of the environment
            reward: Reward for the current step
            terminated: Whether the episode is finished (done flag)
            truncated: Whether the episode was truncated
            info: Additional information for debugging
        """

        vec = np.asarray(action, dtype=np.int32).ravel()
        if vec.size != len(self.action_space):
            raise ValueError("Action must be a flat vector of length 800 (400 sensor-controller pairs)")
        pairs = vec.reshape(int(len(self.action_space)/2), 2)
                # 2️⃣ drop padding rows equal to (0, 0)
        mask_non_padding = ~((pairs[:, 0] == 0) & (pairs[:, 1] == 0))
        pairs = pairs[mask_non_padding]
                # 3️⃣ keep only legal edges  sensor∈{5..316}  →  controller∈{1..4}
        is_sensor      = (pairs[:, 0] >= 5) & (pairs[:, 0] <= 316)
        is_controller  = (pairs[:, 1] >= 1) & (pairs[:, 1] <= 4)
        legal_mask     = is_sensor & is_controller
        pairs          = pairs[legal_mask]
                # 4️⃣ deduplicate while preserving order, cap at 400
        unique_pairs, seen = [], set()
        for s_id, c_id in pairs:
            pair = (int(s_id), int(c_id))
            if pair in seen:
                continue
            seen.add(pair)
            unique_pairs.append(pair)
        print(f"UNIQUE ACTIONS: {len(unique_pairs)}" )  
        for sensor_id, controller_id in unique_pairs:
            try:
                self.add_new_regular_task(sensor_id, controller_id)
            except Exception:
                raise ValueError(f"Invalid sensor ID {sensor_id} or controller ID {controller_id} in action")
                pass      # silently skip impossible combinations

        max_steps = 1000  # Safety limit to prevent infinite loops
        step_count = 0
        for _ in range(20):
            self.time_step()
        #if self.scenario == "light":
        #    self.lights_scenario("2", importance=5)
        if self.scenario == "complex_light":
            raise ValueError("complex_light scenario isny+t posible because room ids are wrong")
            #self.complex_light_scenario(["room8", "room10","room11","room3","room4","room5","room6","room7","room13","room14","room15","room16","room17","room18"], 4)
        elif self.scenario == "multi_sensor_scenario":
            self.multi_sensor_scenario("2")
        while step_count < max_steps:
            # Check if there are any non-info tasks remaining
            non_info_tasks_remain = self.check_non_info_tasks()
                    
            # If no non-info tasks remain, break the loop
            if not non_info_tasks_remain:
                break
                
            # Process time steps
            self.time_step()
            
            # Check for SRAM reaching 70% threshold after each time step, but only for controllers
            for controller_id in usfl_arr.controllers:
                controller = self._vertices_dict.get(str(controller_id))
                if controller and "usedSRAM" in controller and "maxSRAM" in controller and controller["maxSRAM"] > 0:
                    sram_usage_percentage = (controller["usedSRAM"] / controller["maxSRAM"]) * 100
                    if sram_usage_percentage >= 70:
                        ##print(f"⚠️ SRAM usage at controller {controller_id} has reached {sram_usage_percentage:.1f}% of maximum capacity. Terminating step.")
                        return self.constant_obs, -100, False, True, {"error": "SRAM usage threshold exceeded"}
                    
            step_count += 1
        
        # Return the standardconstant_obs0 gym environment outputs
        terminated = True  # Episode ends when all non-info tasks are complete
        truncated = False  # Episode was not truncated
        info = {
            "time": self.time,
            "num_tasks": self.task_id - 1,
            "steps_taken": step_count
        }
        ##print(F"TOTAL LOCAL CONFIRMATIONS: {self.local_confirmation_counter}")
        ##print(F"TOTAL NON-LOCAL CONFIRMATIONS: {self.non_local_confirmation_counter}")
        return self.constant_obs, self.reward, terminated, truncated, info

    def check_non_info_tasks(self):
        # Check vertices
        for vertex in self._vertices_dict.values(): # Use optimized iteration
            for task in vertex.get("tasks", []):
                task_name = task.get("task_name", "UNKNOWN_NAME")
                if task_name != "info_task":
                    return True # Early exit

        # Check connections (iterating the original list is fine here)
        for connection in self.graph.get("connections", []):
            for task in connection.get("tasks", []):
                task_name = task.get("task_name", "UNKNOWN_NAME")
                if task_name != "info_task":
                    return True # Early exit

        # Check confirmation dictionaries
        if self.tasks_awaiting_confirmation:
            return True
        # if self.confirmation_tasks: # Original was commented, implies non-info if active
        #    return True

        return False # No non-info tasks found
    def process_tasks(self):
        """Process tasks in vertices and move them to appropriate connections if possible."""
        for vertex_id_str, vertex in self._vertices_dict.items(): # Iterate using the vertex dictionary
            if not vertex.get("tasks"):
                continue
                
            tasks_to_remove = []
            for task in vertex["tasks"]:
                try:
                    target_vertex_id_str = str(task["target"])
                    
                    if vertex_id_str == target_vertex_id_str:
                        if self.do_task(task, target_vertex_id_str):
                            tasks_to_remove.append(task)
                        continue
                        
                    # Use the optimized function, passing `self` (the GraphEnv instance)
                    next_hop_id_str = usfl_func.get_next_hop(self, vertex_id_str, target_vertex_id_str)
                    
                    if next_hop_id_str is None:
                        ##print(f"Warning: No next hop found for task {task.get('task_id', 'N/A')} from {vertex_id_str} to {target_vertex_id_str}")
                        continue 

                    next_hop_id_str = str(next_hop_id_str) 

                    # Find the connection to the next hop
                    # Using _connections_lookup which stores lists of connections for a pair
                    connection_key = frozenset({vertex_id_str, next_hop_id_str})
                    connections_to_next_hop = self._connections_lookup.get(connection_key)

                    if connections_to_next_hop: # Check if list is not empty
                        # For simplicity, pick the first connection if multiple exist (e.g., parallel links)
                        # More sophisticated logic could be added here if needed (e.g., load balancing)
                        connection = connections_to_next_hop[0]
                        max_tasks = 10 
                        if len(connection.get("tasks", [])) < max_tasks:
                            tasks_to_remove.append(task)
                            connection["tasks"].append(task)
                            #if(task["task_name"] != "info_task"):
                            #    ##print(f" Moving task {task['task_id']} from {vertex_id_str} to {next_hop_id_str}")
                        # else: 
                            # ##print(f"Connection full for task {task['task_id']} from {vertex_id_str} to {next_hop_id_str}")
                    else:
                        pass
                        ##print(f"Warning: Connection object not found between {vertex_id_str} and {next_hop_id_str} for task {task.get('task_id', 'N/A')}")
                            
                except Exception as e:
                    #print(f"Error processing task {task.get('task_id', 'N/A')} at vertex {vertex_id_str}: {e}")
                    continue 
                    
            for task_to_remove in tasks_to_remove: 
                if task_to_remove in vertex["tasks"]:
                    vertex["tasks"].remove(task_to_remove)
                    if "usedSRAM" in vertex:
                        vertex["usedSRAM"] = max(0, vertex["usedSRAM"] - task_to_remove.get("sram_usage", 0))

    def create_info_task(self, location, target_controller, importance=7, task_size=50, sram_usage=2000):
# Get sensor type
        source_vertex = self.find_edge_vertex(str(location))
        if not source_vertex:
            raise ValueError(f"Source vertex {location} not found")
            
        # Get only relevant parameters based on sensor type
        sensor_type = source_vertex["label"]
        filtered_info = {}
        
        if sensor_type == "temperature":
            filtered_info = {
                "temperature": source_vertex.get("temperature"),
                "humidity": source_vertex.get("humidity"),
                "pressure": source_vertex.get("pressure")
            }
        elif sensor_type == "light":
            filtered_info = {
                "brightness": source_vertex.get("brightness"),
                "color_temp": source_vertex.get("color_temp"),
                "isOn": source_vertex.get("isOn")
            }
        elif sensor_type == "LiDAR":
            filtered_info = {
                "distance": source_vertex.get("distance"),
                "angle": source_vertex.get("angle"),
                "intensity": source_vertex.get("intensity")
            }
        elif sensor_type == "CO2":
            filtered_info = {
                "co2": source_vertex.get("co2")
            }
        elif sensor_type == "movement":
            filtered_info = {
                "movement": source_vertex.get("movement"),
                "speed": source_vertex.get("speed"),
                "direction": source_vertex.get("direction")
            }
        elif sensor_type == "noise":
            filtered_info = {
                "decibel": source_vertex.get("decibel"),
                "noise_floor": source_vertex.get("noise_floor"),
                "peak": source_vertex.get("peak")
            }

        task = {
            "task_id": self.task_id,
            "starting_sensor": location,
            "task_location": location,
            "task_name": "info_task",
            "target": target_controller,
            "importance": importance,
            "start_time": self.time,
            "task_size": task_size, 
            "sram_usage": sram_usage,
            "info": filtered_info
        }
        self.task_id += 1
        return task

    def create_confirmation_task(self, requester_task, target_controller_id, actual_sensor_id_to_check, condition, importance=8):
        """Create a task that checks sensor info and confirms if a condition is met.
        
        Args:
            requester_task: The task that needs confirmation
            target_controller_id: ID of the controller where the check should happen
            actual_sensor_id_to_check: ID of the sensor whose data needs checking
            condition: Dict with {"type": "less_than"/"greater_than", "value": number, "comparing parameter": "param_name"}
            importance: Priority level
        """
        task = {
            "task_id": self.task_id,
            "task_name": "confirmation_task",
            "task_location": requester_task["task_location"],
            "target": target_controller_id, # Target is the controller
            "sensor_id": str(actual_sensor_id_to_check),  # The actual sensor ID to check
            "importance": importance,
            "start_time": self.time,
            "task_size": 100,  # Small size for confirmation tasks
            "sram_usage": 1000,
            "requester_task_id": requester_task["task_id"],
            "condition": condition
        }
        self.task_id += 1
        return task

    def create_confirmation_answer_task(self, confirmation_task, is_confirmed):
        """Create a task that sends confirmation results back."""        
        task_location = confirmation_task["target"]  # Start from controller where check happened
        target = confirmation_task["task_location"]  # Send back to original controller/location
        
        answer_task = {
            "task_id": self.task_id,
            "task_name": "confirmation_answer",
            "task_location": task_location,
            "target": target,
            "importance": confirmation_task["importance"],
            "start_time": self.time,
            "task_size": 100,
            "sram_usage": 1000,
            "requester_task_id": confirmation_task["requester_task_id"], # ID of original task (e.g., control_sensor)
            "answered_request_id": confirmation_task["task_id"], # ID of the confirmation_task itself
            "confirmation_result": is_confirmed
        }
        self.task_id += 1
        return answer_task

    def create_task_with_confirmation_needed(self,task,confirmations):
        #confirmations example:
        #confirmations = [
        #    {
        #        "sensor_id": "sensor_1",
        #        "condition": {
        #            "type": "less_than",
        #            "value": 50
        #        },
        #        "importance": 8
        #    }
        #]
        task_with_confirmation_needed = {
            "task": task,
            "confirmations": confirmations,
        }
        return task_with_confirmation_needed
        

    def tasks_autocreation(self):
        """Process all automatic task creation based on descriptions."""
        descriptions = usfl_arr.descriptions_for_regular_tasks
        created_tasks = []
        
        for description in descriptions:
            if(self.time % description["frequency"] != 0):
                continue
            matching_vertices = []
            for vertex in self._vertices_dict.values(): 
                if vertex["label"] == description["label"]:
                    if description["specific_ids"]:
                        if str(vertex["id"]) in [str(id_val) for id_val in description["specific_ids"]]: 
                            matching_vertices.append(vertex)
                    else:
                        matching_vertices.append(vertex)
            
            for vertex in matching_vertices:
                target = description["target"]
                if target is None and vertex.get("closest_controller") and isinstance(vertex["closest_controller"], dict) and "id" in vertex["closest_controller"]:
                    target = vertex["closest_controller"]["id"]
                elif target is None:
                    # Use the optimized function, passing `self`
                    target = usfl_func.find_closest_controller(self, str(vertex["id"]))
                
                task = self.create_info_task(str(vertex["id"]), target, description["importance"], description["task_size"], description["sram_usage"])
                
                if "tasks" not in vertex:
                    vertex["tasks"] = []
                    
                if "usedSRAM" in vertex and "maxSRAM" in vertex:
                    if vertex["usedSRAM"] + task["sram_usage"] > vertex["maxSRAM"]:
                        ##print(f"Warning: Not enough SRAM in vertex {vertex['id']} for task {task['task_id']}")
                        continue    
                    vertex["usedSRAM"] = vertex.get("usedSRAM", 0) + task["sram_usage"]
                
                vertex["tasks"].append(task)
                created_tasks.append(task)
        
        return created_tasks
    
    def isController(self, id: Union[str, int]) -> bool:
        vertex = self.find_edge_vertex(str(id))
        if vertex and vertex.get("label") == "controller":
            return True
        return False

    def calculate_task_reward(self, task):
        if task.get("task_name") == "info_task":
            return 0
        if task.get("task_name") == "confirmation_task" or task.get("task_name") == "confirmation_answer":
            self.non_local_confirmation_counter += 0.5
        """Calculate reward for completing a task based on speed and type.
        
        Args:
            task: The completed task
            
        Returns:
            float: The calculated reward
        """
        # Base reward for different task types
        rewards_multiplier = {
            "control_sensor": 1,
            "confirmation_task": 0.25,
            "confirmation_answer": 0.25
        }
        reward_multiplier = rewards_multiplier.get(task.get("task_name", ""), 0.1)
        
        # Speed bonus calculation
       
        time_taken = self.time - task["start_time"]

        reward = reward_multiplier * usfl_func.reward_calculator(time_taken) * ((task.get("importance",100)/10))
        return reward

    def update_reward(self, task):
        """Update the environment reward when a task is completed."""
        reward = self.calculate_task_reward(task)
        self.reward += reward
        return reward

    def do_task(self, task, target):
        target = str(target)
        target_vertex = self.find_edge_vertex(target)
        
        if not target_vertex:
            raise NameError(f"❌ Task {task} failed - Invalid target")
            
        # If this is the final destination (target sensor)
        if str(task["target"]) == target:
            # For confirmation tasks, check the condition and create answer task
            if task.get("task_name") == "confirmation_task":
                # Check if this is a controller with sensor info (our case)
                if target_vertex.get("label") == "controller" and "sensors_info" in target_vertex:
                    sensor_id = str(task.get("sensor_id", ""))
                    
                    # If no specific sensor_id is provided, use the one from the condition
                    if not sensor_id and "condition" in task:
                        sensor_id = str(task["condition"].get("sensor_id", ""))
                    
                    if sensor_id not in target_vertex["sensors_info"]:
                        ##print(f"❌ Confirmation task {task['task_id']} failed - No sensor info for sensor {sensor_id} in controller {target}")
                        return False
                    
                    condition = task["condition"]
                    comparing_parameter = condition.get("comparing parameter", "")
                    
                    if not comparing_parameter:
                        ##print(f"❌ Confirmation task {task['task_id']} failed - No comparing parameter specified")
                        return False
                    
                    # Get sensor info
                    sensor_info = target_vertex["sensors_info"][sensor_id]["info"]
                    
                    if comparing_parameter not in sensor_info:
                        ##print(f"❌ Confirmation task {task['task_id']} failed - Parameter {comparing_parameter} not found in sensor {sensor_id} info")
                        return False
                    
                    value = sensor_info[comparing_parameter]
                    
                    # Check condition
                    condition_met = False
                    if condition["type"] == "less_than":
                        condition_met = value < condition["value"]
                    elif condition["type"] == "greater_than":
                        condition_met = value > condition["value"]
                    
                    reward = self.update_reward(task)
                    ##print(f"🔍 Confirmation task {task['task_id']} checked {comparing_parameter}={value} against condition {condition} - Result: {condition_met} - Reward: {reward:.3f}")
                    
                    # Create answer task
                    answer_task = self.create_confirmation_answer_task(task, condition_met)
                    
                    # Find the controller that sent the confirmation task
                    controller = self.find_edge_vertex(task["task_location"])
                    if controller and "tasks" not in controller:
                        controller["tasks"] = []
                    if controller:
                        controller["tasks"].append(answer_task)
                        ##print(f"📤 Created confirmation answer task {answer_task['task_id']} and sent to controller {task['task_location']}")
                    return True
                    
                # Original code for non-controller confirmation targets
                elif "value" not in target_vertex:
                    ##print(f"❌ Confirmation task {task['task_id']} failed - No value in target sensor {target}")
                    return False
                else:
                    condition = task["condition"]
                    value = target_vertex["value"]
                    
                    condition_met = False
                    if condition["type"] == "less_than":
                        condition_met = value < condition["value"]
                    elif condition["type"] == "greater_than":
                        condition_met = value > condition["value"]
                    
                    reward = self.update_reward(task)
                    ##print(f"🔍 Confirmation task {task['task_id']} checked value {value} against condition {condition} - Result: {condition_met} - Reward: {reward:.3f}")
                    
                    # Create answer task
                    answer_task = self.create_confirmation_answer_task(task, condition_met)
                    
                    # Find the controller that sent the confirmation task
                    controller = self.find_edge_vertex(task["task_location"])
                    if controller and "tasks" not in controller:
                        controller["tasks"] = []
                    if controller:
                        controller["tasks"].append(answer_task)
                        ##print(f"📤 Created confirmation answer task {answer_task['task_id']} and sent to controller {task['task_location']}")
                    return True

            # For info tasks, store info by sensor ID    
            elif task.get("task_name") == "info_task":
                # ... existing code ...
                if "sensors_info" not in target_vertex:
                    target_vertex["sensors_info"] = {}
                source_id = str(task.get("starting_sensor"))
                target_vertex["sensors_info"][source_id] = {
                    "info": task.get("info", {}),
                    "timestamp": self.time,
                }
                reward = self.update_reward(task)
                return True

            # For confirmation answer tasks, process them
            elif task.get("task_name") == "confirmation_answer":
                # ... existing code ...
                reward = self.update_reward(task)
                ##print(f"📥 Processing confirmation answer task {task['task_id']} - Reward: {reward:.3f}")
                self.handle_confirmation_answer(task)
                # Remove the confirmation task since we have its answer
                if task["task_id"] in self.confirmation_tasks:
                    del self.confirmation_tasks[task["task_id"]]
                return True

            # For light control tasks
            elif task.get("task_name") == "control_sensor":
                # ... existing code ...
                if "parameters_to_change" in task:
                    params = task["parameters_to_change"]
                    # Make sure we're finding the correct target light vertex again
                    light_vertex = self.find_edge_vertex(str(task["target"]))
                   
                    
                    if not light_vertex:
                        ##print(f"❌ Light control task {task['task_id']} failed - Cannot find target light {task['target']}")
                        return False
                    # Update the light's parameters
                    for param, value in params.items():
                        light_vertex[param] = value
                reward = self.update_reward(task)
                ##print(f"💡 Light control task {task['task_id']} completed - Set brightness to {params.get('brightness', 'N/A')} on light {target} - Reward: {reward:.3f}")
                return True

            # For regular tasks that reached their target
            else:
                ##print(f"❌❌❌ERRORRRR: Task {task['task_id']} completed at target {target} (BUT NOTHING HAPPENED)")
                ##print(f"Task details: {task}")
                return True
        # For intermediate nodes (including controllers)
        if "tasks" not in target_vertex:
            target_vertex["tasks"] = []
            
        # Check and update SRAM for controllers
        if "usedSRAM" in target_vertex and "maxSRAM" in target_vertex:
            if target_vertex["usedSRAM"] + task.get("sram_usage", 0) > target_vertex["maxSRAM"]:
                ##print(f"❌ Task {task['task_id']} failed - Not enough SRAM at node {target}. Environment stopping.")
                return False
            target_vertex["usedSRAM"] += task.get("sram_usage", 0)
            
        target_vertex["tasks"].append(task)
        #if(task["task_name"] != "info_task"):
        #    ##print(f"📦 Task {task['task_id']} delivered to intermediate node {target}")
        task["task_location"] = target  # Update task location
        return True

    def handle_task_with_confirmation(self, task_with_confirmation):
        """Handle a task that needs confirmation from other sensors.
        
        The confirmation is done by checking controller information rather than 
        directly accessing sensors.
        """
        task = task_with_confirmation["task"]
        confirmations = task_with_confirmation["confirmations"]
        
        self.tasks_awaiting_confirmation[task["task_id"]] = {
            "task": task,
            "confirmations_needed": len(confirmations),
            "confirmations_received": 0,
            "all_confirmed": True,
            "remote_check_initiated": False # Initialize flag for remote checks
        }
        
        # Get the current controller
        current_controller = self.find_edge_vertex(task["task_location"])
        if not current_controller or current_controller.get("label") != "controller":
            raise ValueError(f"Task location {task['task_location']} is not a controller")
        
        # Process each confirmation request
        for confirmation in confirmations:
            sensor_id = confirmation["sensor_id"]
            sensor_id_str = str(sensor_id)
            
            # First check if the current controller has the needed info
            if "sensors_info" in current_controller and sensor_id_str in current_controller["sensors_info"]:
                self.local_confirmation_counter+=1
                self.reward += usfl_func.reward_calculator(0) * 2 * 0.25 
                self._process_local_confirmation(task, current_controller, sensor_id_str, confirmation)
                continue
                
            # If not, find the closest controller that might have the info
            # Mark that a remote check is needed BEFORE creating the task
            self.tasks_awaiting_confirmation[task["task_id"]]["remote_check_initiated"] = True
            closest_controller = self._find_controller_with_sensor_info(current_controller["id"], sensor_id_str)
            
            if not closest_controller:
                # Reset flag if we fail to find a controller? Or let it fail? Let it fail for now.
                raise ValueError(f"No controller has information for sensor {sensor_id}")
                
            # Create confirmation task to send to the closest controller
            # Corrected the arguments passed to create_confirmation_task
            conf_task = self.create_confirmation_task(
                requester_task=task,
                target_controller_id=closest_controller["id"],  # Send to the controller that has the info
                actual_sensor_id_to_check=sensor_id,  # Pass the actual sensor ID
                condition=confirmation["condition"], # Pass the condition dictionary
                importance=confirmation.get("importance", 8)
            )
            
            # Store confirmation task
            self.confirmation_tasks[conf_task["task_id"]] = conf_task
            
            # Add the task to the current controller's tasks
            if "tasks" not in current_controller:
                current_controller["tasks"] = []
            current_controller["tasks"].append(conf_task)
    
    def _process_local_confirmation(self, task, controller, sensor_id, confirmation):
        """Process a confirmation locally when the controller already has the sensor info."""
        # Get the sensor info from the controller
        sensor_info = controller["sensors_info"][sensor_id]["info"]
        
        # Get the value based on comparing parameter from the condition
        comparing_parameter = confirmation["condition"].get("comparing parameter", "")
        value = None
        
        if comparing_parameter in sensor_info:
            value = sensor_info[comparing_parameter]
        else:
            raise ValueError(f"Sensor {sensor_id} does not have the parameter {comparing_parameter}")
            
        if value is None:
            raise ValueError(f"Could not find a valid value in sensor {sensor_id} info")
        
        # Check if condition is met
        condition_met = False
        condition = confirmation["condition"]
        if condition["type"] == "less_than":
            condition_met = value < condition["value"]
        elif condition["type"] == "greater_than":
            condition_met = value > condition["value"]
        
        # Update the task awaiting confirmation
        if task["task_id"] in self.tasks_awaiting_confirmation:
            awaiting_task = self.tasks_awaiting_confirmation[task["task_id"]]
            awaiting_task["confirmations_received"] += 1
            
            if not condition_met:
                awaiting_task["all_confirmed"] = False
            # If all confirmations are received, process the task
            if awaiting_task["confirmations_received"] >= awaiting_task["confirmations_needed"]:
                self._finalize_task_with_confirmation(awaiting_task)
                
    def _find_controller_with_sensor_info(self, current_controller_id, sensor_id):
        """Find the closest controller that has information about the specified sensor."""
        controllers = [v for v in self._vertices_dict.values() if v["label"] == "controller"]
        
        
        controllers_with_info = []
        for controller in controllers:
            controller_id_str = str(controller["id"])
            sensor_keys = list(controller.get("sensors_info", {}).keys())
            
            if controller_id_str == str(current_controller_id):
                ##print(f"Skipping current controller {controller_id_str}")
                continue
                
            if "sensors_info" in controller and str(sensor_id) in controller["sensors_info"]:
                ##print(f"Found sensor {sensor_id} info in controller {controller_id_str}")
                try:
                    # Use the optimized function, passing `self`
                    path = usfl_func.find_fastest_path_optimized(self, str(current_controller_id), controller_id_str)
                    if path:
                        distance = len(path) - 1
                        controllers_with_info.append((controller, distance))
                        ##print(f"Path found to controller {controller_id_str} with distance {distance}")
                    else:
                        pass
                        ##print(f"No path found to controller {controller_id_str}")
                except Exception as e:
                    ##print(f"Error finding path to controller {controller_id_str}: {e}")
                    continue
        
        if controllers_with_info:
            closest_controller, distance = min(controllers_with_info, key=lambda x: x[1])
           
            return closest_controller
        
        ##print(f"WARNING: No controller found with information for sensor {sensor_id}")
        return None
        
    def _finalize_task_with_confirmation(self, awaiting_task):
        """Finalize a task that was waiting for confirmations."""
        if awaiting_task["all_confirmed"]:
            # All confirmations successful, proceed with original task
            original_task = awaiting_task["task"]
            
            # Check if all confirmations were local
            #if not awaiting_task.get("remote_check_initiated", False):
            #     original_task["locally_confirmed"] = True
            #     ##print(f"  Task {original_task['task_id']} confirmed locally.")
            #     # Add +2 reward for local confirmation
            #     self.reward += usfl_func.reward_calculator(0) * 2 * 0.8
            #     ##print(f"  Awarded +2 bonus for local confirmation of task {original_task['task_id']}. Current total reward: {self.reward}")
#
            # Get the controller that should execute the task - this is where the task should be located,
            # not on the target vertex directly
            controller_vertex = self.find_edge_vertex(str(original_task["task_location"]))
            
            if controller_vertex:
                controller_vertex["tasks"].append(original_task)
                ##print(f"✅ All confirmations passed for task {original_task['task_id']}, proceeding with task on controller {controller_vertex['id']}")
            else:
                pass
                ##print(f"❌ Cannot find controller vertex {original_task['task_location']} for task {original_task['task_id']}")
        else:
            pass
            ##print(f"❌ Confirmations failed for task {awaiting_task['task']['task_id']}, canceling task")
            
        # Clean up the awaiting confirmation task
        del self.tasks_awaiting_confirmation[awaiting_task["task"]["task_id"]]

    def handle_confirmation_answer(self, answer_task):
        """Process a confirmation answer task.
        
        Args:
            answer_task: Dict containing the confirmation answer
        """
        requester_task_id = answer_task["requester_task_id"]
            
        if requester_task_id in self.tasks_awaiting_confirmation:
            awaiting_task = self.tasks_awaiting_confirmation[requester_task_id]
            awaiting_task["confirmations_received"] += 1
            
            # If confirmation failed, mark the task as failed
            if not answer_task["confirmation_result"]:
                awaiting_task["all_confirmed"] = False
            
            # If we've received all confirmations
            if awaiting_task["confirmations_received"] >= awaiting_task["confirmations_needed"]:
                original_task = awaiting_task["task"] # Get the original task

                if awaiting_task["all_confirmed"]:
                    # Find the controller where the answer arrived (which is the target of the answer task)
                    controller_vertex = self.find_edge_vertex(str(answer_task["target"]))

                    if (controller_vertex):
                        if "tasks" not in controller_vertex:
                            controller_vertex["tasks"] = []
                        # Add the original task to the CONTROLLER's queue to start its journey
                        # Ensure the task's location is updated to the controller
                        original_task["task_location"] = controller_vertex["id"]
                        controller_vertex["tasks"].append(original_task)
                        
                        # Check if this is a task with the trigger_all_lights flag
                        if original_task.get("trigger_all_lights") == True:
                            ##print(f"✨ Task {original_task['task_id']} was confirmed and has trigger_all_lights flag. Triggering lights scenario.")
                            # Trigger the lights scenario with all lights
                            controller_id = str(controller_vertex["id"])
                            self.lights_scenario(controller_id, importance=7,starting_time=original_task["start_time"])
                            self.turn_off_ventilation_scenario(controller_id, importance=4,starting_time=original_task["start_time"])
                            self.lock_doors_scenario(controller_id, importance=9,starting_time=original_task["start_time"])
                    else:
                        pass
                        # This shouldn't happen if the graph is consistent
                        ##print(f"❌ DEBUG: Could not find controller vertex {answer_task['target']} to queue confirmed task {original_task['task_id']}")

                else:
                    pass
                    ##print(f"❌ DEBUG: Task {requester_task_id} failed confirmation checks. Canceling.")
                    # Optionally, add logic here if failed tasks need specific handling
                
                # Clean up the awaiting task regardless of success/failure, now that all confirmations are in
                del self.tasks_awaiting_confirmation[requester_task_id]

                # Clean up the original confirmation request task from the tracking dictionary
                # Use the answered_request_id (which is the ID of the confirmation_task itself)
                confirmation_task_id_to_remove = answer_task.get("answered_request_id")
                if confirmation_task_id_to_remove and confirmation_task_id_to_remove in self.confirmation_tasks:
                     del self.confirmation_tasks[confirmation_task_id_to_remove]
                     
                elif confirmation_task_id_to_remove:
                    pass
                    ##print(f"⚠️ Tried to remove confirmation task {confirmation_task_id_to_remove}, but it wasn't found in the dictionary.") # Added warning
                else:
                    pass
                    ##print(f"⚠️ Confirmation answer task {answer_task['task_id']} is missing 'answered_request_id'. Cannot clean up confirmation_tasks.") # Added warning

        # The removal of the answer task itself happens in process_confirmations

    def process_confirmations(self):
        """Process all confirmation-related tasks in the environment."""
        # First process any confirmation answers
        for vertex in self._vertices_dict.values(): # Use optimized iteration
            tasks_to_remove = []
            for task in vertex.get("tasks", []):
                if task.get("task_name") == "confirmation_answer":
                    self.handle_confirmation_answer(task)
                    tasks_to_remove.append(task)
            
            # Remove processed confirmation answer tasks
            for task in tasks_to_remove:
                vertex["tasks"].remove(task)
                if task["task_id"] in self.confirmation_tasks:
                    del self.confirmation_tasks[task["task_id"]]

    def time_step(self):

        """Process one time step in the environment."""
        ##print(f"\n\n=== Time step {self.time} ===", flush=True)
        # Process rest of time step
        created_tasks = self.tasks_autocreation()
        
        # Only ##print information about non-info tasks
        #if created_tasks:
        #    non_info_tasks = [task for task in created_tasks if task.get("task_name") != "info_task"]
        #    if non_info_tasks:
        #        print(f"\n🔍 Created {len(non_info_tasks)} non-info tasks:")
        #        for task in non_info_tasks:
        #            
        #            print(f"Task {task['task_id']}: {task['task_name']} to {task['target']}")

        self.process_confirmations()
        self.process_tasks()
        
        for connection in self.graph.get("connections", []):
            # Calculate available bandwidth for this connection
            # Simplified calculation
            available_bandwidth = connection["speed"] * self.time_step_in_ms
            available_bandwidth += connection["reserved_from_previous_step"]
            
            # Reset reserved for the start of this step's processing
            connection["reserved_from_previous_step"] = 0
            
            if not connection["tasks"]:
                continue
            
            # Sort tasks by importance to ensure high priority tasks are processed first
            connection["tasks"].sort(key=lambda x: x.get("importance", 0), reverse=True)
            
            tasks_to_remove = []
            for task in connection["tasks"]:
                # DEBUG ##print added previously
                

                if available_bandwidth >= task["task_size"]:
                    # Determine the target vertex for do_task
                    source_node = str(connection["source"])
                    target_node = str(connection["target"])
                    task_origin = str(task["task_location"]) # Where the task was before entering the connection

                    destination_node = None
                    if task_origin == source_node:
                        destination_node = target_node
                    elif task_origin == target_node:
                        destination_node = source_node
                    else:
                        # Handle error or unexpected state
                        ##print(f"Error: Task {task['task_id']} on connection {source_node}-{target_node} has unexpected origin {task_origin}. Skipping task.")
                        continue # Skip task if destination is unclear

                    if self.do_task(task, destination_node): # Pass the calculated destination
                        tasks_to_remove.append(task)
                        available_bandwidth -= task["task_size"]
                        
                    

                else:
                    connection["reserved_from_previous_step"] = available_bandwidth # Save remaining BW for next step
                    break # Stop processing tasks for this connection
            
            # Remove processed tasks
            for task in tasks_to_remove:
                if task in connection["tasks"]:
                    connection["tasks"].remove(task)
        
        self.time = self.time + self.time_step_in_ms
        return True
    def close(self):
        """
        Close the environment.
        """
        pass

        
    def find_edge_vertex(self, edge_id: Union[str, int]) -> Optional[Dict]:
        """Find vertex by ID using the lookup dictionary."""
        return self._vertices_dict.get(str(edge_id))

    def complex_light_scenario(self, room_ids: List[Union[str, int]], starting_controller_id: Union[str, int]):
        """
        Turns on lights in specified rooms if temperature < 1000 and LiDAR > 0.

        Args:
            room_ids: List of room names (e.g., ["room8", "room9"]) or IDs.
            starting_controller_id: The ID of the controller initiating the scenario.
        """
        starting_controller_id = str(starting_controller_id)
        controller_vertex = self.find_edge_vertex(starting_controller_id)
        if not controller_vertex or controller_vertex.get("label") != "controller":
            ##print(f"Error: Starting ID {starting_controller_id} is not a valid controller.")
            return

        if "tasks" not in controller_vertex:
            controller_vertex["tasks"] = []

        for room_name in room_ids:
            if (room_name not in usfl_arr.rooms):
                raise Exception(f"Warning: Room '{room_name}' not found in usefull_arrays. Skipping.")
                continue

            room_data = usfl_arr.rooms[room_name]
            light_ids = room_data.get("lights", [])
            temp_sensor_id = room_data.get("temperature")
            lidar_sensor_id = room_data.get("LiDAR")

            if not light_ids:
                ##print(f"Warning: No lights found for room '{room_name}'. Skipping.")
                continue
            if not temp_sensor_id:
                ##print(f"Warning: No temperature sensor found for room '{room_name}'. Cannot apply condition.")
                # Decide if you want to skip or proceed without temp check
                continue
            if not lidar_sensor_id:
                ##print(f"Warning: No LiDAR sensor found for room '{room_name}'. Cannot apply condition.")
                # Decide if you want to skip or proceed without lidar check
                continue

            # --- Create Confirmation Conditions ---
            confirmations = []
            # Temperature condition
            confirmations.append({
                "sensor_id": str(temp_sensor_id),
                "condition": {
                    "type": "less_than",
                    "value": 1000,
                    "comparing parameter": "temperature"
                },
                "importance": 8 # High importance for conditions
            })
            # LiDAR condition (assuming LiDAR info has a 'people_count' or similar key)
            # IMPORTANT: Adjust "comparing parameter" based on actual LiDAR info structure
            confirmations.append({
                "sensor_id": str(lidar_sensor_id),
                "condition": {
                    "type": "greater_than",
                    "value": 0,
                    "comparing parameter": "distance" # Placeholder: Change to the actual parameter name for people count
                },
                "importance": 8
            })

            # --- Create Light Control Tasks with Confirmation ---
            for light_id in light_ids:
                light_id_str = str(light_id)
                # Create the main task (control_sensor)
                light_task = {
                    "task_id": self.task_id,
                    "task_name": "control_sensor",
                    "task_location": starting_controller_id, # Task originates from the controller
                    "target": light_id_str,
                    "importance": 5, # Example importance
                    "start_time": self.time,
                    "task_size": 100,
                    "sram_usage": 40,
                    "parameters_to_change": {
                        "brightness": 8, # Example: Set brightness to 8
                        "isOn": 1,       # Turn the light on
                    }
                }
                self.task_id += 1

                # Bundle task with confirmations
                task_with_confirmation = self.create_task_with_confirmation_needed(
                    task=light_task,
                    confirmations=confirmations
                )

                # Handle the task (initiate confirmation process)
                try:
                    # Ensure the task starts its confirmation journey from the controller
                    # The handle_task_with_confirmation expects the task's location to be the controller
                    self.handle_task_with_confirmation(task_with_confirmation)
                    ##print(f"Initiated light control task {light_task['task_id']} for light {light_id_str} in {room_name} with confirmations.")
                except Exception as e:
                    ##print(f"Error initiating task for light {light_id_str} in {room_name}: {e}")
                    # Decrement task_id if creation failed before handling
                    self.task_id -= 1


    def scenarios_for_rooms(self,room_ids,starting_controller):
        processed_room_ids = []
        if isinstance(room_ids, (list, tuple)):
            for rid in room_ids:
                try:
                    processed_room_ids.append(int(rid))
                except ValueError:
                    pass
                    ##print(f"Warning: Could not convert room_id '{rid}' to int. Skipping.")
        elif isinstance(room_ids, (str, int)): # Handle single room_id case
            try:
                processed_room_ids.append(int(room_ids))
            except ValueError:
                pass
                ##print(f"Warning: Could not convert room_id '{room_ids}' to int. Skipping.")
        else:
            ##print(f"Warning: room_ids type {type(room_ids)} not supported. Skipping.")
            return

        rooms_data_source = usfl_arr.rooms # Assuming usfl_arr.rooms is the correct dict

        for room_name, room_data in rooms_data_source.items():
            if not room_name.startswith("room"):
                continue
            try:
                room_number = int(room_name.replace("room", ""))
                if room_number in processed_room_ids:
                    ##print(f"\\nRoom: {room_name}")
                    for sensor_type, value in room_data.items():
                        pass
                        ##print(f"  {sensor_type}: {value}")
            except ValueError:
                pass
                ##print(f"Warning: Could not parse room number from '{room_name}'. Skipping room.")
            # Removed 'break' statement that was here to process all relevant rooms
            
    def add_new_regular_task(self,sensor_id,target):
        label = self.find_edge_vertex(str(sensor_id))["label"]
        usfl_arr.add_new_regular_task(target, sensor_id, label)

    def lights_scenario(self, starting_vertex,starting_time = -1, targets = [], importance=7):
        if(starting_time == -1):
            starting_time = self.time
        """Create tasks for controlling lights from a starting vertex.
        
        Args:
            starting_vertex: ID of the vertex initiating the tasks
            targets: List of target light IDs to control
            importance: Priority level of the tasks
                        
        Returns:
            Tuple[List[Dict], bool]: List of created tasks and success flag
        """
        if not targets:
            # Take first 40 light IDs from usfl_arr.light
            if len(usfl_arr.light) > 40:
                targets = usfl_arr.light[:40]
            else:
                targets = usfl_arr.light
        starting_vertex = str(starting_vertex)
        
        task_size = 100  # 100 bits
        SRAMusage = 40  # 40 b 
        
        # Find source vertex and verify SRAM capacity
        source_vertex = self.find_edge_vertex(starting_vertex)
        if not source_vertex:
            raise ValueError(f"Source vertex {starting_vertex} not found in graph vertices")
            
        if "usedSRAM" in source_vertex and "maxSRAM" in source_vertex:
            if source_vertex["usedSRAM"] + SRAMusage * len(targets) > source_vertex["maxSRAM"]:
                raise ValueError(f'Not enough SRAM in source vertex {starting_vertex} to create tasks: {source_vertex["usedSRAM"]}')

        tasks = []
        for target in targets:
            task = {
                "task_id": self.task_id,
                "task_name": "control_sensor",
                "task_location": starting_vertex,
                "target": target,
                "importance": importance,
                "start_time": starting_time,
                "task_size": task_size,
                "sram_usage": SRAMusage,
                "parameters_to_change": {
                    "brightness": 7,
                    "isOn": 1,
                    "duration": 10,
                }
            }
            self.task_id += 1
            tasks.append(task)

                    # Add tasks to the source vertex
        if "tasks" not in source_vertex:
            source_vertex["tasks"] = []
        source_vertex["tasks"].extend(tasks)
        source_vertex["usedSRAM"] = source_vertex.get("usedSRAM", 0) + SRAMusage * len(tasks)

        return tasks, True  # Return both tasks and success flag

    def multi_sensor_scenario(self, starting_controller_id):
        """
        Creates a scenario where one light task is sent with confirmations from ALL sensors
        of each type (CO2, movement, LiDAR, noise), and when that task is confirmed, 
        it triggers turning on all lights.
        
        Args:
            starting_controller_id: ID of the controller initiating the scenario
        """
        starting_controller_id = str(starting_controller_id)
        controller_vertex = self.find_edge_vertex(starting_controller_id)
        
        if not controller_vertex or controller_vertex.get("label") != "controller":
            ##print(f"Error: Starting ID {starting_controller_id} is not a valid controller.")
            return
        # We'll select one light to control initially
        light_id = usfl_arr.light[0] if usfl_arr.light else None
        if not light_id:
            ##print("No lights available for scenario")
            raise ValueError("No lights available for scenario")
            
        # Create confirmations from different sensor types with always-true conditions
        confirmations = []
        
        # Add ALL CO2 sensors confirmations (always true condition)
        if usfl_arr.CO2:
            for co2_sensor_id in usfl_arr.CO2:
                confirmations.append({
                    "sensor_id": str(co2_sensor_id),
                    "condition": {
                        "type": "greater_than",
                        "value": -1,  # Always true
                        "comparing parameter": "co2"
                    },
                    "importance": 8
                })
            ##print(f"Added confirmations from {len(usfl_arr.CO2)} CO2 sensors")
            
        # Add ALL movement sensors confirmations (always true condition)
        if usfl_arr.movement:
            for movement_sensor_id in usfl_arr.movement:
                confirmations.append({
                    "sensor_id": str(movement_sensor_id),
                    "condition": {
                        "type": "greater_than", 
                        "value": -1,  # Always true
                        "comparing parameter": "movement"
                    },
                    "importance": 8
                })
            ##print(f"Added confirmations from {len(usfl_arr.movement)} movement sensors")
            
        # Add ALL LiDAR sensors confirmations (always true condition)
        if usfl_arr.LiDAR:
            for lidar_sensor_id in usfl_arr.LiDAR:
                confirmations.append({
                    "sensor_id": str(lidar_sensor_id),
                    "condition": {
                        "type": "greater_than",
                        "value": -1,  # Always true
                        "comparing parameter": "distance"
                    },
                    "importance": 8
                })
            ##print(f"Added confirmations from {len(usfl_arr.LiDAR)} LiDAR sensors")
            
        # Add ALL noise sensors confirmations (always true condition)
        if usfl_arr.noise:
            for noise_sensor_id in usfl_arr.noise:
                confirmations.append({
                    "sensor_id": str(noise_sensor_id),
                    "condition": {
                        "type": "greater_than",
                        "value": -1,  # Always true
                        "comparing parameter": "decibel"
                    },
                    "importance": 8
                })
            ##print(f"Added confirmations from {len(usfl_arr.noise)} noise sensors")
            
        ##print(f"Total confirmations required: {len(confirmations)} from all sensors")
            
        # Create initial light control task that requires confirmations
        light_task = {
            "task_id": self.task_id,
            "task_name": "control_sensor",
            "task_location": starting_controller_id,
            "target": str(light_id),
            "importance": 5,
            "start_time": self.time,
            "task_size": 100,
            "sram_usage": 4000,
            "parameters_to_change": {
                "brightness": 5,  # Medium brightness
                "isOn": 1,        # Turn the light on
            },
            "trigger_all_lights": True  # Custom flag to trigger all lights scenario
        }
        self.task_id += 1
        
        # Bundle task with confirmations
        task_with_confirmation = self.create_task_with_confirmation_needed(
            task=light_task,
            confirmations=confirmations
        )
        
        # Handle the task (initiate confirmation process)
        try:
            self.handle_task_with_confirmation(task_with_confirmation)
            ##print(f"Initiated light control task {light_task['task_id']} for light {light_id} with confirmations from ALL sensors.")
            ##print(f"When this task is confirmed, all lights will turn on.")
        except Exception as e:
            ##print(f"Error initiating multi_sensor_scenario: {e}")
            # Decrement task_id if creation failed before handling
            self.task_id -= 1

    def lock_doors_scenario(self, starting_vertex,starting_time = -1, targets = [], importance=9):
        if(starting_time == -1):
            starting_time = self.time
        """Create tasks for locking doors from a starting vertex.
        
        Args:
            starting_vertex: ID of the vertex initiating the tasks
            targets: List of target door IDs to lock (if empty, all doors will be locked)
            importance: Priority level of the tasks
                        
        Returns:
            Tuple[List[Dict], bool]: List of created tasks and success flag
        """
        if not targets:
            # Take all door IDs from usfl_arr.door
            targets = usfl_arr.door
        starting_vertex = str(starting_vertex)
        
        task_size = 100  # 100 bits
        SRAMusage = 30  # 30 bytes 
        
        # Find source vertex and verify SRAM capacity
        source_vertex = self.find_edge_vertex(starting_vertex)
        if not source_vertex:
            raise ValueError(f"Source vertex {starting_vertex} not found in graph vertices")
            
        if "usedSRAM" in source_vertex and "maxSRAM" in source_vertex:
            if source_vertex["usedSRAM"] + SRAMusage * len(targets) > source_vertex["maxSRAM"]:
                raise ValueError(f'Not enough SRAM in source vertex {starting_vertex} to create tasks: {source_vertex["usedSRAM"]}')

        tasks = []
        for target in targets:
            task = {
                "task_id": self.task_id,
                "task_name": "control_sensor",
                "task_location": starting_vertex,
                "target": target,
                "importance": importance,
                "start_time": starting_time,
                "task_size": task_size,
                "sram_usage": SRAMusage,
                "parameters_to_change": {
                    "locked": 1,  # 1 means locked
                    "secure_mode": 1,  # Enable secure mode
                }
            }
            self.task_id += 1
            tasks.append(task)
            ##print(f"Created lock task for door {target}")

        # Add tasks to the source vertex
        if "tasks" not in source_vertex:
            source_vertex["tasks"] = []
        source_vertex["tasks"].extend(tasks)
        source_vertex["usedSRAM"] = source_vertex.get("usedSRAM", 0) + SRAMusage * len(tasks)
        
        ##print(f"✅ Created {len(tasks)} door locking tasks from controller {starting_vertex}")
        return tasks, True  # Return both tasks and success flag

    def turn_off_ventilation_scenario(self, starting_vertex,starting_time = -1, targets = [], importance=7):
        if(starting_time == -1):
            starting_time = self.time
        """Create tasks for turning off ventilation systems from a starting vertex.
        
        Args:
            starting_vertex: ID of the vertex initiating the tasks
            targets: List of target ventilation IDs to turn off (if empty, all ventilations will be turned off)
            importance: Priority level of the tasks
                        
        Returns:
            Tuple[List[Dict], bool]: List of created tasks and success flag
        """
        if not targets:
            # Take all ventilation IDs from usfl_arr.ventelation
            targets = usfl_arr.ventelation
        starting_vertex = str(starting_vertex)
        
        task_size = 100  # 100 bits
        SRAMusage = 25  # 25 bytes 
        
        # Find source vertex and verify SRAM capacity
        source_vertex = self.find_edge_vertex(starting_vertex)
        if not source_vertex:
            raise ValueError(f"Source vertex {starting_vertex} not found in graph vertices")
            
        if "usedSRAM" in source_vertex and "maxSRAM" in source_vertex:
            if source_vertex["usedSRAM"] + SRAMusage * len(targets) > source_vertex["maxSRAM"]:
                raise ValueError(f'Not enough SRAM in source vertex {starting_vertex} to create tasks: {source_vertex["usedSRAM"]}')

        tasks = []
        for target in targets:
            task = {
                "task_id": self.task_id,
                "task_name": "control_sensor",
                "task_location": starting_vertex,
                "target": target,
                "importance": importance,
                "start_time": starting_time,
                "task_size": task_size,
                "sram_usage": SRAMusage,
                "parameters_to_change": {
                    "isOn": 0,         # Turn ventilation off
                    "fan_speed": 0,     # Set fan speed to 0
                    "maintenance_mode": 1  # Enable maintenance mode
                }
            }
            self.task_id += 1
            tasks.append(task)
            ##print(f"Created task to turn off ventilation {target}")

        # Add tasks to the source vertex
        if "tasks" not in source_vertex:
            source_vertex["tasks"] = []
        source_vertex["tasks"].extend(tasks)
        source_vertex["usedSRAM"] = source_vertex.get("usedSRAM", 0) + SRAMusage * len(tasks)
        
        ##print(f"✅ Created {len(tasks)} ventilation shutdown tasks from controller {starting_vertex}")
        return tasks, True  # Return both tasks and success flag
