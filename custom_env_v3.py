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
    

    def __init__(self, descriptions_for_regular_tasks, json_path="graph_output.json", time_step=10,scenario = "light"):
        super(GraphEnv, self).__init__()
        self.json_path = json_path
        self.time_step_in_ms = time_step
        self.descriptions_for_regular_tasks = descriptions_for_regular_tasks
        
        print("Loading graph from JSON file...", flush=True)
        # Load the graph from the JSON file
        with open(json_path, 'r') as f:
            self.graph = json.load(f)
        print(f"Graph loaded successfully: {len(self.graph.get('vertices', []))} vertices, {len(self.graph.get('connections', []))} connections", flush=True)
            
        print("Initializing graph connections...", flush=True)
        # Initialize graph components
        for connection in self.graph.get("connections", []):
            connection["tasks"] = []
            connection["reserved_from_previous_step"] = 0
        print("Connections initialized", flush=True)
            
        print("Initializing vertices...", flush=True)
        vertex_count = 0
        # Initialize vertices
        for vertex in self.graph.get("vertices", []):
            vertex_count += 1
            if vertex_count % 100 == 0:  # Print status every 100 vertices
                print(f"Processed {vertex_count} vertices so far...", flush=True)
                
            vertex["tasks"] = []
            if vertex.get("label") == "controller":
                vertex["sensors_info"] = {}
            elif vertex.get("label") != "controller":
                try:
                    closest_controller = usfl_func.find_closest_controller(self.graph, str(vertex["id"]))
                    path = usfl_func.find_fastest_path(self.graph, str(vertex["id"]), closest_controller)
                    distance = len(path) - 1
                    vertex["closest_controller"] = {
                        "id": closest_controller,
                        "distance": distance
                    }
                except ValueError as e:
                    print(f"Warning: Could not find closest controller for sensor {vertex['id']}: {e}")
                    vertex["closest_controller"] = None
        print("All vertices initialized", flush=True)
    
        # Initialize environment variables
        self.scenario = scenario
        self.time = 0
        self.task_id = 1
        self.reward = 0
        self.tasks_awaiting_confirmation = {}
        self.confirmation_tasks = {}
        self.graph_copy = copy.deepcopy(self.graph)  # Use deepcopy instead of shallow copy
        self.observation = {
            "graph": self.graph,
        }
        print("Environment variables initialized", flush=True)
        
        # Simplified action space for testing: List of [sensor_id, target] pairs
        # Each element is a tuple of two integers
        self.action_space = spaces.Box(
            low=0,
            high=1000,
            shape=(10, 2),  # Up to 10 tasks per step, each with [sensor_id, target]
            dtype=np.int32
        )
        
        # Simplified observation space
        self.observation_space = spaces.Dict({
            "graph": spaces.Dict({
                "vertices": spaces.Sequence(
                    spaces.Dict({
                        "id": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
                        "label": spaces.Text(max_length=20),
                        "tasks": spaces.Sequence(
                            spaces.Dict({
                                "task_id": spaces.Box(low=0, high=10000, shape=(), dtype=np.int32),
                                "task_name": spaces.Text(max_length=50),
                                "target": spaces.Box(low=0, high=1000, shape=(), dtype=np.int32),
                                "importance": spaces.Box(low=0, high=10, shape=(), dtype=np.int32)
                            })
                        )
                    })
                ),
                "connections": spaces.Sequence(
                    spaces.Dict({
                        "source": spaces.Discrete(1000),
                        "target": spaces.Discrete(1000),
                        "speed": spaces.Box(low=0, high=1000000, shape=(), dtype=np.float32)
                    })
                )
            })
        })
    
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
        usfl_arr.reset_regular_tasks()
        # Reset task-related dictionaries
        self.tasks_awaiting_confirmation = {}
        self.confirmation_tasks = {}
        
        # Use a deep copy of the original graph
        self.graph = copy.deepcopy(self.graph_copy)
        
        # Update observation
        self.observation = {
            "graph": self.graph,
        }
        
        return self.observation, {}

    def step(self, matrix):
        """Process an Nx2 matrix where each row contains a sensor ID and target, and advance the environment until all non-info tasks are complete.
        
        Args:
            matrix: An Nx2 numpy array or list of lists where each row contains:
                   - Column 0: sensor ID
                   - Column 1: target ID
                   

        Returns:
            observation: Current state of the environment
            reward: Reward for the current step
            done: Whether the episode is finished
            info: Additional information for debugging
        """

        if(len(matrix) == 0):
            matrix = []
        elif len(matrix[0]) != 2:
            raise ValueError("Matrix must have exactly 2 columns (sensor_id, target)")
            
        # Process all tasks in the matrix
        for i in range(len(matrix)):
            sensor_id = matrix[i][0]
            target = matrix[i][1]
            try:
                self.add_new_regular_task(sensor_id, target)
            except Exception as e:
                print(f"Warning: Failed to add task for sensor {sensor_id}: {e}")

        max_steps = 1000  # Safety limit to prevent infinite loops
        step_count = 0
        #for _ in range(10):
        #    self.time_step()
        #if self.scenario == "light":
            #self.lights_scenario("4", importance=5)
        while step_count < max_steps:
            # Check if there are any non-info tasks remaining
            non_info_tasks_remain = self.check_non_info_tasks()
                    
            # Check confirmation tasks
            if self.confirmation_tasks or self.tasks_awaiting_confirmation:
                non_info_tasks_remain = True
                
            # If no non-info tasks remain, break the loop
            if not non_info_tasks_remain:
                break
                
            # Process time steps
            self.time_step()
            step_count += 1
        
        # Return the standard gym environment outputs
        done = True  # Episode ends when all non-info tasks are complete
        info = {
            "time": self.time,
            "num_tasks": self.task_id - 1,
            "steps_taken": step_count
        }
        rewardq = self.reward
        return self.observation, rewardq, done, info

    def check_non_info_tasks(self):
        found_non_info = False # Flag to track if we found any

        # Check vertices
        for vertex in self.graph["vertices"]:
            for task in vertex.get("tasks", []):
                task_name = task.get("task_name", "UNKNOWN_NAME")
                if task_name != "info_task":
                    
                    found_non_info = True
                    # Don't return immediately, let's see if there are others

        # Check connections
        for connection in self.graph["connections"]:
            for task in connection.get("tasks", []):
                task_name = task.get("task_name", "UNKNOWN_NAME")
                if task_name != "info_task":
                   
                    found_non_info = True
                    # Don't return immediately

        # Check confirmation dictionaries
        if self.tasks_awaiting_confirmation:

            found_non_info = True
        if self.confirmation_tasks:

            found_non_info = True
        return found_non_info # Return the collected status
    def process_tasks(self):
        """Process tasks in vertices and move them to appropriate connections if possible."""
        for vertex in self.graph["vertices"]:
            if not vertex.get("tasks"):
                continue
                
            tasks_to_remove = []
            for task in vertex["tasks"]:
                try:
                    current_vertex = str(vertex["id"])
                    target_vertex = str(task["target"])
                    
                    if current_vertex == target_vertex:
                        # If this is the target, process the task
                        if self.do_task(task, target_vertex):
                            tasks_to_remove.append(task)
                        continue
                        
                    # Find path to target - simplified: always use get_next_hop
                    # Removed the complex path-through-controller logic for confirmation tasks
                    next_hop = usfl_func.get_next_hop(self.graph, current_vertex, target_vertex)
                    
                    # Find the connection to the next hop
                    for connection in self.graph["connections"]:
                        if ((str(connection["source"]) == current_vertex and str(connection["target"]) == str(next_hop)) or
                            (str(connection["source"]) == str(next_hop) and str(connection["target"]) == current_vertex)):
                            
                            # Check if connection has space for more tasks
                            # Simplified max_tasks calculation (example)
                            max_tasks = 10 # Example: Allow up to 10 tasks per connection step
                            # max_tasks = (self.time_step_in_ms + 30) / 4 # Original calculation
                            if len(connection.get("tasks", [])) < max_tasks:
                                tasks_to_remove.append(task)
                                connection["tasks"].append(task)
                                if(task["task_name"] != "info_task"):
                                    print(f" Moving task {task['task_id']} from {current_vertex} to {next_hop}")
                            break
                            
                except Exception as e:
                    # Print a more informative error message
                    print(f"Error processing task {task.get('task_id', 'N/A')} at vertex {current_vertex}: {e}")
                    # Optionally, decide if the task should be removed or retried
                    # tasks_to_remove.append(task) # Example: remove failing task
                    continue # Continue to the next task
                    
            # Remove processed tasks from vertex
            for task in tasks_to_remove:
                # Check if task is still in the list before removing
                if task in vertex["tasks"]:
                    vertex["tasks"].remove(task)
                    if "usedSRAM" in vertex:
                        vertex["usedSRAM"] = max(0, vertex["usedSRAM"] - task.get("sram_usage", 0))

    def create_info_task(self, location, target_controller, importance=7, task_size=100, sram_usage=4000):
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
            "task_size": 50,  # Small size for confirmation tasks
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
            "task_size": 50,
            "sram_usage": 1000,
            "requester_task_id": confirmation_task["requester_task_id"], # ID of original task (e.g., control_light)
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
            # Get vertices with matching label
            matching_vertices = []
            for vertex in self.graph["vertices"]:
                if vertex["label"] == description["label"]:
                    if description["specific_ids"]:
                        if str(vertex["id"]) in [str(id) for id in description["specific_ids"]]:
                            matching_vertices.append(vertex)
                    else:
                        matching_vertices.append(vertex)
            
            # For each matching vertex, create a task
            for vertex in matching_vertices:
                # If target is not specified, use pre-calculated closest controller
                target = description["target"]
                if target is None and vertex.get("closest_controller"):
                    target = vertex["closest_controller"]["id"]
                elif target is None:
                    try:
                        target = self.handle_closest_controller(str(vertex["id"]))
                    except ValueError as e:
                        raise Exception(f"Error finding controller for vertex {vertex['id']}: {e}")
                
                # Remove debug print for vertex 4's info tasks
                
                # Create and add task
                task = self.create_info_task(vertex["id"], target, description["importance"], description["task_size"], description["sram_usage"])
                
                # Add task to vertex
                if "tasks" not in vertex:
                    vertex["tasks"] = []
                    
                # Check SRAM capacity if applicable
                if "usedSRAM" in vertex and "maxSRAM" in vertex:
                    if vertex["usedSRAM"] + task["sram_usage"] > vertex["maxSRAM"]:
                        raise (f"Warning: Not enough SRAM in vertex {vertex['id']} for task")
                        continue    
                    vertex["usedSRAM"] = vertex.get("usedSRAM", 0) + task["sram_usage"]
                
                vertex["tasks"].append(task)
                created_tasks.append(task)
                self.task_id += 1
        
        return created_tasks

    def handle_closest_controller(self, vertex_id):
        vertex = self.find_edge_vertex(str(vertex_id))
        if not vertex:
            raise ValueError(f"Vertex {vertex_id} not found")
            
        if "closest_controller" not in vertex:
            vertex["closest_controller"] = usfl_func.find_closest_controller(self.graph, str(vertex["id"]))
            
        return vertex["closest_controller"]

    def lights_scenario(self, starting_vertex, targets = [], importance=4):
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
                "task_name": "control_light",
                "task_location": starting_vertex,
                "target": target,
                "importance": importance,
                "start_time": self.time,
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

    def isController(self, id: Union[str, int]) -> bool:
        vertex = self.find_edge_vertex(str(id))
        if vertex and vertex.get("label") == "controller":
            return True
        return False

    def calculate_task_reward(self, task):
        if task.get("task_name") == "info_task":
            return 0
        """Calculate reward for completing a task based on speed and type.
        
        Args:
            task: The completed task
            
        Returns:
            float: The calculated reward
        """
        # Base reward for different task types
        base_rewards = {
            "control_light": 0.2,
            "confirmation_task": 0.08,
            "confirmation_answer": 0.08
        }
        
        base_reward = base_rewards.get(task.get("task_name", ""), 0.1)
        
        # Speed bonus calculation
        time_taken = self.time - task["start_time"]
        print(task["start_time"], self.time, time_taken)
        # Maximum expected time is 1000ms, minimum is 10ms
        normalized_time = max(0, min(1, 1 - ((time_taken - 10) / 990)))
        speed_bonus = normalized_time * 0.15  # Up to 0. bonus for speed
        bonus_for_importance = task.get("importance", 4) / 100  # Scale importance to a 0-1 range
        return base_reward + speed_bonus + bonus_for_importance

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
                    
                    # Debug output
                    print(f"Processing confirmation task {task['task_id']} for sensor {sensor_id} on controller {target}")
                    
                    if sensor_id not in target_vertex["sensors_info"]:
                        print(f"❌ Confirmation task {task['task_id']} failed - No sensor info for sensor {sensor_id} in controller {target}")
                        return False
                    
                    condition = task["condition"]
                    comparing_parameter = condition.get("comparing parameter", "")
                    
                    if not comparing_parameter:
                        print(f"❌ Confirmation task {task['task_id']} failed - No comparing parameter specified")
                        return False
                    
                    # Get sensor info
                    sensor_info = target_vertex["sensors_info"][sensor_id]["info"]
                    
                    if comparing_parameter not in sensor_info:
                        print(f"❌ Confirmation task {task['task_id']} failed - Parameter {comparing_parameter} not found in sensor {sensor_id} info")
                        return False
                    
                    value = sensor_info[comparing_parameter]
                    
                    # Check condition
                    condition_met = False
                    if condition["type"] == "less_than":
                        condition_met = value < condition["value"]
                    elif condition["type"] == "greater_than":
                        condition_met = value > condition["value"]
                    
                    reward = self.update_reward(task)
                    print(f"🔍 Confirmation task {task['task_id']} checked {comparing_parameter}={value} against condition {condition} - Result: {condition_met} - Reward: {reward:.3f}")
                    
                    # Create answer task
                    answer_task = self.create_confirmation_answer_task(task, condition_met)
                    
                    # Find the controller that sent the confirmation task
                    controller = self.find_edge_vertex(task["task_location"])
                    if controller and "tasks" not in controller:
                        controller["tasks"] = []
                    if controller:
                        controller["tasks"].append(answer_task)
                        print(f"📤 Created confirmation answer task {answer_task['task_id']} and sent to controller {task['task_location']}")
                    return True
                    
                # Original code for non-controller confirmation targets
                elif "value" not in target_vertex:
                    print(f"❌ Confirmation task {task['task_id']} failed - No value in target sensor {target}")
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
                    print(f"🔍 Confirmation task {task['task_id']} checked value {value} against condition {condition} - Result: {condition_met} - Reward: {reward:.3f}")
                    
                    # Create answer task
                    answer_task = self.create_confirmation_answer_task(task, condition_met)
                    
                    # Find the controller that sent the confirmation task
                    controller = self.find_edge_vertex(task["task_location"])
                    if controller and "tasks" not in controller:
                        controller["tasks"] = []
                    if controller:
                        controller["tasks"].append(answer_task)
                        print(f"📤 Created confirmation answer task {answer_task['task_id']} and sent to controller {task['task_location']}")
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
                print(f"📥 Processing confirmation answer task {task['task_id']} - Reward: {reward:.3f}")
                self.handle_confirmation_answer(task)
                # Remove the confirmation task since we have its answer
                if task["task_id"] in self.confirmation_tasks:
                    del self.confirmation_tasks[task["task_id"]]
                return True

            # For light control tasks
            elif task.get("task_name") == "control_light":
                # ... existing code ...
                if "parameters_to_change" in task:
                    params = task["parameters_to_change"]
                    # Make sure we're finding the correct target light vertex again
                    light_vertex = self.find_edge_vertex(str(task["target"]))
                    print(f"🔍 DEBUG: Light control task {task['task_id']} - Found light vertex: {light_vertex['id'] if light_vertex else 'NOT FOUND'}")
                    
                    if not light_vertex:
                        print(f"❌ Light control task {task['task_id']} failed - Cannot find target light {task['target']}")
                        return False
                        
                    # Print current values before update
                    print(f"🔍 DEBUG: Before update - Light {task['target']} state: isOn={light_vertex.get('isOn', 'N/A')}, brightness={light_vertex.get('brightness', 'N/A')}")
                    
                    # Update the light's parameters
                    for param, value in params.items():
                        light_vertex[param] = value
                        print(f"🔍 DEBUG: Setting {param}={value} on light {task['target']}")
                        
                    # Verify the update worked
                    print(f"🔍 DEBUG: After update - Light {task['target']} state: isOn={light_vertex.get('isOn', 'N/A')}, brightness={light_vertex.get('brightness', 'N/A')}")
                    
                reward = self.update_reward(task)
                print(f"💡 Light control task {task['task_id']} completed - Set brightness to {params.get('brightness', 'N/A')} on light {target} - Reward: {reward:.3f}")
                return True

            # For regular tasks that reached their target
            else:
                print(f"❌❌❌ERRORRRR: Task {task['task_id']} completed at target {target} (BUT NOTHING HAPPENED)")
                return True

        # For intermediate nodes (including controllers)
        if "tasks" not in target_vertex:
            target_vertex["tasks"] = []
            
        # Check and update SRAM for controllers
        if "usedSRAM" in target_vertex and "maxSRAM" in target_vertex:
            if target_vertex["usedSRAM"] + task.get("sram_usage", 0) > target_vertex["maxSRAM"]:
                raise ValueError(f"❌ Task {task['task_id']} failed - Not enough SRAM at node {target}")
                return False
            target_vertex["usedSRAM"] += task.get("sram_usage", 0)
            
        target_vertex["tasks"].append(task)
        if(task["task_name"] != "info_task"):
            print(f"📦 Task {task['task_id']} delivered to intermediate node {target}")
        task["task_location"] = target  # Update task location
        return True

    def handle_task_with_confirmation(self, task_with_confirmation):
        """Handle a task that needs confirmation from other sensors.
        
        The confirmation is done by checking controller information rather than 
        directly accessing sensors.
        """
        task = task_with_confirmation["task"]
        confirmations = task_with_confirmation["confirmations"]
        
        # Store task in awaiting confirmations
        self.tasks_awaiting_confirmation[task["task_id"]] = {
            "task": task,
            "confirmations_needed": len(confirmations),
            "confirmations_received": 0,
            "all_confirmed": True
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
                # Process confirmation locally
                self._process_local_confirmation(task, current_controller, sensor_id_str, confirmation)
                continue
                
            # If not, find the closest controller that might have the info
            closest_controller = self._find_controller_with_sensor_info(current_controller["id"], sensor_id_str)
            
            if not closest_controller:
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
            print(f"📋 Created confirmation task {conf_task['task_id']} to check sensor {sensor_id} info at controller {closest_controller['id']}")
    
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
                
            print(f"🔍 Local confirmation for task {task['task_id']} checked {comparing_parameter} value {value} against condition {condition} - Result: {condition_met}")
            
            # If all confirmations are received, process the task
            if awaiting_task["confirmations_received"] >= awaiting_task["confirmations_needed"]:
                self._finalize_task_with_confirmation(awaiting_task)
                
    def _find_controller_with_sensor_info(self, current_controller_id, sensor_id):
        """Find the closest controller that has information about the specified sensor."""
        controllers = [v for v in self.graph["vertices"] if v["label"] == "controller"]
        
        # Debug output
        print(f"Looking for controller with info for sensor {sensor_id}")
        print(f"Current controller ID: {current_controller_id}")
        
        # Check each controller for the sensor info
        controllers_with_info = []
        for controller in controllers:
            controller_id = str(controller["id"])
            
            # Debug output for each controller's sensor info
            sensor_keys = list(controller.get("sensors_info", {}).keys())
            print(f"Controller {controller_id} has sensors: {sensor_keys}")
            
            # Skip current controller as we already checked it in the calling method
            if controller_id == str(current_controller_id):
                print(f"Skipping current controller {controller_id}")
                continue
                
            if "sensors_info" in controller and str(sensor_id) in controller["sensors_info"]:
                print(f"Found sensor {sensor_id} info in controller {controller_id}")
                # Calculate distance from current controller to this one
                try:
                    # Using find_fastest_path instead of find_path
                    path = usfl_func.find_fastest_path(self.graph, str(current_controller_id), controller_id)
                    if path:
                        distance = len(path) - 1
                        controllers_with_info.append((controller, distance))
                        print(f"Path found to controller {controller_id} with distance {distance}")
                    else:
                        print(f"No path found to controller {controller_id}")
                except Exception as e:
                    print(f"Error finding path to controller {controller_id}: {e}")
                    continue
        
        # Return the closest controller with the info
        if controllers_with_info:
            closest_controller, distance = min(controllers_with_info, key=lambda x: x[1])
            print(f"Selected closest controller: {closest_controller['id']} with distance {distance}")
            return closest_controller
        
        print(f"WARNING: No controller found with information for sensor {sensor_id}")
        return None
        
    def _finalize_task_with_confirmation(self, awaiting_task):
        """Finalize a task that was waiting for confirmations."""
        if awaiting_task["all_confirmed"]:
            # All confirmations successful, proceed with original task
            original_task = awaiting_task["task"]
            
            # Get the controller that should execute the task - this is where the task should be located,
            # not on the target vertex directly
            controller_vertex = self.find_edge_vertex(str(original_task["task_location"]))
            
            if controller_vertex:
                controller_vertex["tasks"].append(original_task)
                print(f"✅ All confirmations passed for task {original_task['task_id']}, proceeding with task on controller {controller_vertex['id']}")
            else:
                print(f"❌ Cannot find controller vertex {original_task['task_location']} for task {original_task['task_id']}")
        else:
            print(f"❌ Confirmations failed for task {awaiting_task['task']['task_id']}, canceling task")
            
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
                    print(f"🔄 DEBUG: Confirmed task {original_task['task_id']} - target: {original_task['target']}, params: {original_task.get('parameters_to_change', 'N/A')}")

                    # Find the controller where the answer arrived (which is the target of the answer task)
                    controller_vertex = self.find_edge_vertex(str(answer_task["target"]))

                    if controller_vertex:
                        if "tasks" not in controller_vertex:
                            controller_vertex["tasks"] = []
                        # Add the original task to the CONTROLLER's queue to start its journey
                        # Ensure the task's location is updated to the controller
                        original_task["task_location"] = controller_vertex["id"]
                        controller_vertex["tasks"].append(original_task)
                        print(f"✅ DEBUG: Added confirmed task {original_task['task_id']} to controller {controller_vertex['id']} tasks queue for delivery to {original_task['target']}")
                    else:
                        # This shouldn't happen if the graph is consistent
                        print(f"❌ DEBUG: Could not find controller vertex {answer_task['target']} to queue confirmed task {original_task['task_id']}")

                else:
                    print(f"❌ DEBUG: Task {requester_task_id} failed confirmation checks. Canceling.")
                    # Optionally, add logic here if failed tasks need specific handling
                
                # Clean up the awaiting task regardless of success/failure, now that all confirmations are in
                del self.tasks_awaiting_confirmation[requester_task_id]

                # Clean up the original confirmation request task from the tracking dictionary
                # Use the answered_request_id (which is the ID of the confirmation_task itself)
                confirmation_task_id_to_remove = answer_task.get("answered_request_id")
                if confirmation_task_id_to_remove and confirmation_task_id_to_remove in self.confirmation_tasks:
                     del self.confirmation_tasks[confirmation_task_id_to_remove]
                     print(f"🗑️ Removed confirmation task {confirmation_task_id_to_remove} from tracking dictionary.") # Added log
                elif confirmation_task_id_to_remove:
                    print(f"⚠️ Tried to remove confirmation task {confirmation_task_id_to_remove}, but it wasn't found in the dictionary.") # Added warning
                else:
                    print(f"⚠️ Confirmation answer task {answer_task['task_id']} is missing 'answered_request_id'. Cannot clean up confirmation_tasks.") # Added warning

        # The removal of the answer task itself happens in process_confirmations

    def process_confirmations(self):
        """Process all confirmation-related tasks in the environment."""
        # First process any confirmation answers
        for vertex in self.graph["vertices"]:
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
        print(f"\n\n=== Time step {self.time} ===", flush=True)
        # Process rest of time step
        created_tasks = self.tasks_autocreation()
        
        # Only print information about non-info tasks
        if created_tasks:
            non_info_tasks = [task for task in created_tasks if task.get("task_name") != "info_task"]
            if non_info_tasks:
                print(f"\n🔍 Created {len(non_info_tasks)} non-info tasks:")
                for task in non_info_tasks:
                    print(f"Task {task['task_id']}: {task['task_name']} to {task['target']}")

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
                # DEBUG print added previously
                

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
                        print(f"Error: Task {task['task_id']} on connection {source_node}-{target_node} has unexpected origin {task_origin}. Skipping task.")
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

    def render(self, mode="human"):
        """
        Render the environment (not implemented).
        """
        pass
    
    def close(self):
        """
        Close the environment.
        """
        pass

        
    def find_edge_vertex(self, edge_id):
        """Find vertex by edge ID.        """
        for vertex in self.graph["vertices"]:
            if str(vertex["id"]) == str(edge_id):
                return vertex
        return None
    
    def scenarios_for_rooms(self,room_ids,starting_controller):
        for room_id in room_ids:
            room_id = int(room_id)
        rooms = usfl_arr.rooms
        # Print sample room information
        for room_name, room_data in rooms.items():
            if(not room_name.__contains__("room")):
                continue
            if int(room_name.replace("room", "")) in room_ids:
                print(f"\nRoom: {room_name}")
            for sensor_type, value in room_data.items():
                print(f"{sensor_type}: {value}")
            break  # Just print the first room for now
            
    def add_new_regular_task(self,sensor_id,target):
        label = self.find_edge_vertex(str(sensor_id))["label"]
        usfl_arr.add_new_regular_task(target, sensor_id, label)
        
    

if __name__ == "__main__": 
    print("\n=== Testing Cross-Controller Confirmation with sensor 41 ===\n")
    env = GraphEnv(usfl_arr.descriptions_for_regular_tasks)

    # Reset the environment
    obs, info = env.reset()
    print("Environment initialized and reset")
    
    # First run time steps to populate sensor data
    print("Running initial time steps to populate sensor data...")
    for _ in range(30):  # Run more steps to ensure sensor data is collected
        env.time_step()
    
    # The specific controllers and sensor we want to use
    controller_3_id = "3"
    controller_4_id = "4"
    target_sensor_id = "41"
    target_light = "76"
    
    controller_3 = env.find_edge_vertex(controller_3_id)
    controller_4 = env.find_edge_vertex(controller_4_id)
    
    if not controller_3 or not controller_4:
        print(f"Error: Could not find controller 3 or 4")
        exit(1)
    
    print(f"Using controller {controller_3_id} as source (task creator)")
    print(f"Using controller {controller_4_id} as confirmation provider")
    
    # Make sure controller 4 has sensors_info
    if "sensors_info" not in controller_4:
        controller_4["sensors_info"] = {}
    
    # Always create/update the sensor info for sensor 41 on controller 4
    # This ensures we have the data needed for the confirmation
    controller_4["sensors_info"][target_sensor_id] = {
        "info": {
            "movement": 2.28,  # Using the value from your error message
            "speed": 5
        },
        "timestamp": env.time
    }
    
    # Verify the sensor info was properly set
    if target_sensor_id not in controller_4["sensors_info"]:
        print(f"Error: Failed to create sensor info for {target_sensor_id}")
        exit(1)
        
    # Get the sensor parameter and value that we'll use for confirmation
    sensor_parameter = "movement"
    sensor_value = controller_4["sensors_info"][target_sensor_id]["info"][sensor_parameter]
    
    # Double check that the sensor info is there
    print(f"Verified sensor {target_sensor_id} with {sensor_parameter}={sensor_value} at controller {controller_4_id}")
    
    # Create light control task at controller 3
    control_task = {
        "task_id": env.task_id,
        "task_name": "control_light",
        "task_location": controller_3_id,  # Starting from controller 3
        "target": target_light,
        "importance": 6,
        "start_time": env.time,
        "task_size": 1000,
        "sram_usage": 4000,
        "parameters_to_change": {
            "brightness": 200,
            "isOn": 1,
            "duration": 5,
        }
    }
    env.task_id += 1
    
    # Set confirmation threshold higher than the current value to ensure it passes
    threshold_value = sensor_value + 100  
    
    # Create confirmation requirement for sensor 41 data from controller 4
    confirmation = {
        "sensor_id": target_sensor_id,
        "condition": {
            "comparing parameter": sensor_parameter,
            "type": "less_than",
            "value": threshold_value
        },
        "importance": 8
    }
    
    print(f"Setting up confirmation: Check if {sensor_parameter}={sensor_value} is less than {threshold_value}")
    
    # Create task with confirmation
    task_with_confirmation = env.create_task_with_confirmation_needed(control_task, [confirmation])
    
    # Print the controllers' sensor info for debugging
    print(f"\nController 3 has info for sensors: {list(controller_3.get('sensors_info', {}).keys())}")
    print(f"Controller 4 has info for sensors: {list(controller_4.get('sensors_info', {}).keys())}")
    
    # Handle the task with confirmation
    print("\nProcessing task with confirmation between controllers 3 and 4...")
    try:
        env.handle_task_with_confirmation(task_with_confirmation)
        
        # Use step() to process the environment until completion
        print("\nUsing step() function to process all tasks until completion...")
        obs, reward, done, info = env.step([])
        
        # Check if light was controlled
        light = env.find_edge_vertex(target_light)
        print("\n=== Task Completion Results ===")
        print(f"Environment processed {info['steps_taken']} steps")
        print(f"Total time elapsed: {info['time']}ms")
        print(f"Total reward accumulated: {reward}")
        
        if light:
            print(f"\nTarget light {target_light} final state:")
            print(f"IsOn: {light.get('isOn', 'N/A')}")
            print(f"Brightness: {light.get('brightness', 'N/A')}")
        else:
            print(f"Error: Could not find light {target_light}")
    except ValueError as e:
        print(f"Error during confirmation test: {e}")