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
    

    def __init__(self, descriptions_for_regular_tasks, json_path="graph_output.json", time_step=10):
        super(GraphEnv, self).__init__()
        self.json_path = json_path
        self.time_step_in_ms = time_step
        self.descriptions_for_regular_tasks = descriptions_for_regular_tasks
        
        # Load the graph from the JSON file
        with open(json_path, 'r') as f:
            self.graph = json.load(f)
            
        # Initialize graph components
        for connection in self.graph.get("connections", []):
            connection["tasks"] = []
            connection["reserved_from_previous_step"] = 0
            
        # Initialize vertices
        for vertex in self.graph.get("vertices", []):
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
    
        # Initialize environment variables
        self.time = 0
        self.task_id = 1
        self.reward = 0
        self.tasks_awaiting_confirmation = {}
        self.confirmation_tasks = {}
        self.graph_copy = copy.deepcopy(self.graph)  # Use deepcopy instead of shallow copy
        self.observation = {
            "graph": self.graph,
        }
        
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
        # Validate input matrix
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
        
        while step_count < max_steps:
            # Check if there are any non-info tasks remaining
            non_info_tasks_remain = False
            
            # Check vertices for non-info tasks
            for vertex in self.graph["vertices"]:
                for task in vertex.get("tasks", []):
                    if task.get("task_name") != "info_task":
                        non_info_tasks_remain = True
                        break
                if non_info_tasks_remain:
                    break
                    
            # Check connections for non-info tasks
            for connection in self.graph["connections"]:
                for task in connection.get("tasks", []):
                    if task.get("task_name") != "info_task":
                        non_info_tasks_remain = True
                        break
                if non_info_tasks_remain:
                    break
                    
            # Check confirmation tasks
            if self.confirmation_tasks or self.tasks_awaiting_confirmation:
                non_info_tasks_remain = True
                
            # If no non-info tasks remain, break the loop
            if not non_info_tasks_remain:
                break
                
            # Process time steps
            for i in range(10):
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
        self.reset()
        return self.observation, rewardq, done, info

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
                        
                    # Find path to target through controllers if needed
                    if task.get("task_name") in ["confirmation_task", "confirmation_answer"]:
                        # For confirmation tasks, find path through controllers
                        controllers = [v for v in self.graph["vertices"] if v["label"] == "controller"]
                        paths = []
                        for controller in controllers:
                            try:
                                path1 = usfl_func.find_path(self.graph, current_vertex, str(controller["id"]))
                                path2 = usfl_func.find_path(self.graph, str(controller["id"]), target_vertex)
                                if path1 and path2:
                                    cost = (sum(usfl_func.get_connection_cost(self.graph, path1[i], path1[i+1]) 
                                             for i in range(len(path1)-1)) +
                                          sum(usfl_func.get_connection_cost(self.graph, path2[i], path2[i+1])
                                             for i in range(len(path2)-1)))
                                    paths.append((path1[1], cost))  # Just take next hop from first path
                            except Exception:
                                raise("wtf is this and  why this works, or does it?")
                                continue
                        
                        if paths:
                            # Take the path with lowest cost
                            next_hop, _ = min(paths, key=lambda x: x[1])
                        else:
                            next_hop = usfl_func.get_next_hop(self.graph, current_vertex, target_vertex)
                    else:
                        next_hop = usfl_func.get_next_hop(self.graph, current_vertex, target_vertex)
                    
                    # Find the connection to the next hop
                    for connection in self.graph["connections"]:
                        if ((str(connection["source"]) == current_vertex and str(connection["target"]) == str(next_hop)) or
                            (str(connection["source"]) == str(next_hop) and str(connection["target"]) == current_vertex)):
                            
                            # Check if connection has space for more tasks
                            max_tasks = (self.time_step_in_ms + 30) / 4
                            if len(connection.get("tasks", [])) < max_tasks:
                                tasks_to_remove.append(task)
                                connection["tasks"].append(task)
                                print(f"🚀 Moving task {task['task_id']} from {current_vertex} to {next_hop}")
                            break
                            
                except Exception as e:
                    print(f"Error processing task {task.get('task_id')}: {e}")
                    continue
                    
            # Remove processed tasks from vertex
            for task in tasks_to_remove:
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

    def create_confirmation_task(self, requester_task, sensor_id, condition, importance=8):
        """Create a task that checks sensor info and confirms if a condition is met.
        
        Args:
            requester_task: The task that needs confirmation
            sensor_id: ID of the sensor to check
            condition: Dict with {"type": "less_than"/"greater_than", "value": number}
            importance: Priority level
        """
        task = {
            "task_id": self.task_id,
            "task_name": "confirmation_task",
            "task_location": requester_task["task_location"],
            "target": sensor_id,
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
        task_location = confirmation_task["target"]  # Start from sensor
        target = confirmation_task["task_location"]  # Send back to original controller
        
        answer_task = {
            "task_id": self.task_id,
            "task_name": "confirmation_answer",
            "task_location": task_location,
            "target": target,
            "importance": confirmation_task["importance"],
            "start_time": self.time,
            "task_size": 50,
            "sram_usage": 1000,
            "requester_task_id": confirmation_task["requester_task_id"],
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
                        print(f"Error finding controller for vertex {vertex['id']}: {e}")
                        continue
                
                # Debug print for vertex 4's sensors
                if str(target) == "4":
                    print(f"🔄 Creating task for sensor {vertex['id']} ({vertex['label']}) to send info to controller 4")
                
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
        """Calculate reward for completing a task based on speed and type.
        
        Args:
            task: The completed task
            
        Returns:
            float: The calculated reward
        """
        # Base reward for different task types
        base_rewards = {
            "info_task": 0,
            "control_light": 0.2,
            "confirmation_task": 0.08,
            "confirmation_answer": 0.08
        }
        
        base_reward = base_rewards.get(task.get("task_name", ""), 0.1)
        
        # Speed bonus calculation
        time_taken = self.time - task["start_time"]
        # Maximum expected time is 1000ms, minimum is 10ms
        normalized_time = max(0, min(1, 1 - (time_taken - 10) / 990))
        speed_bonus = normalized_time * 0.1  # Up to 0.1 bonus for speed
        
        return base_reward + speed_bonus

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
        if not target_vertex == target_vertex:
            raise NameError(f"❌ Task {task} failed - we are not at the target")
        if str(task["target"]) == target:
            # For confirmation tasks, check the condition and create answer task
            if task.get("task_name") == "confirmation_task":
                if "value" not in target_vertex:
                    print(f"❌ Confirmation task {task['task_id']} failed - No value in target sensor {target}")
                    return False

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
                if "sensors_info" not in target_vertex:
                    target_vertex["sensors_info"] = {}
                source_id = str(task.get("starting_sensor"))
                target_vertex["sensors_info"][source_id] = {
                    "info": task.get("info", {}),
                    "timestamp": self.time,
                }
                reward = self.update_reward(task)
                print(f"🎯 Info task {task['task_id']} from sensor {source_id} delivered to controller {target} - Reward: {reward:.3f}")
                return True

            # For confirmation answer tasks, process them
            elif task.get("task_name") == "confirmation_answer":
                reward = self.update_reward(task)
                print(f"📥 Processing confirmation answer task {task['task_id']} - Reward: {reward:.3f}")
                self.handle_confirmation_answer(task)
                # Remove the confirmation task since we have its answer
                if task["task_id"] in self.confirmation_tasks:
                    del self.confirmation_tasks[task["task_id"]]
                return True

            # For light control tasks
            elif task.get("task_name") == "control_light":
                if "parameters_to_change" in task:
                    params = task["parameters_to_change"]
                    if "brightness" in params:
                        target_vertex["brightness"] = params["brightness"]
                    if "isOn" in params:
                        target_vertex["isOn"] = params["isOn"]
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
        print(f"📦 Task {task['task_id']} delivered to intermediate node {target}")
        task["task_location"] = target  # Update task location
        return True

    def handle_task_with_confirmation(self, task_with_confirmation):
        """Handle a task that needs confirmation from other sensors."""
        task = task_with_confirmation["task"]
        confirmations = task_with_confirmation["confirmations"]
        
        # Store task in awaiting confirmations
        self.tasks_awaiting_confirmation[task["task_id"]] = {
            "task": task,
            "confirmations_needed": len(confirmations),
            "confirmations_received": 0,
            "all_confirmed": True
        }
        
        # Create and send confirmation tasks
        for confirmation in confirmations:
            sensor_id = confirmation["sensor_id"]
            sensor = self.find_edge_vertex(sensor_id)
            
            # Create confirmation task
            conf_task = self.create_confirmation_task(
                task,
                sensor_id,
                confirmation["condition"],
                confirmation.get("importance", 8)
            )
            
            # Store confirmation task
            self.confirmation_tasks[conf_task["task_id"]] = conf_task
            
            # Find controller to handle the confirmation
            controller = self.find_edge_vertex(task["task_location"])
            if not controller:
                raise ValueError(f"Controller {task['task_location']} not found in graph vertices")
            if controller:
                if "tasks" not in controller:
                    controller["tasks"] = []
                controller["tasks"].append(conf_task)
                print(f"📋 Created confirmation task {conf_task['task_id']} for sensor {sensor_id}")

    def handle_confirmation_answer(self, answer_task):
        """Process a confirmation answer task.
        
        Args:
            answer_task: Dict containing the confirmation answer
        """
        requester_task_id = answer_task["requester_task_id"]
        # Clean up the original confirmation task
        if answer_task["task_id"] in self.confirmation_tasks:
            del self.confirmation_tasks[answer_task["task_id"]]
            
        if requester_task_id in self.tasks_awaiting_confirmation:
            awaiting_task = self.tasks_awaiting_confirmation[requester_task_id]
            awaiting_task["confirmations_received"] += 1
            
            # If confirmation failed, mark the task as failed
            if not answer_task["confirmation_result"]:
                awaiting_task["all_confirmed"] = False
            
            # If we've received all confirmations
            if awaiting_task["confirmations_received"] >= awaiting_task["confirmations_needed"]:
                if awaiting_task["all_confirmed"]:
                    # All confirmations successful, proceed with original task
                    original_task = awaiting_task["task"]
                    target_vertex = self.find_edge_vertex(str(original_task["target"]))
                    if target_vertex and "tasks" not in target_vertex:
                        target_vertex["tasks"] = []
                    if target_vertex:
                        target_vertex["tasks"].append(original_task)
                
                # Clean up all related confirmation tasks
                confirmation_tasks_to_remove = []
                for conf_task_id, conf_task in self.confirmation_tasks.items():
                    if conf_task["requester_task_id"] == requester_task_id:
                        confirmation_tasks_to_remove.append(conf_task_id)
                
                for conf_task_id in confirmation_tasks_to_remove:
                    if conf_task_id in self.confirmation_tasks:
                        del self.confirmation_tasks[conf_task_id]
                
                # Clean up the awaiting confirmation task
                del self.tasks_awaiting_confirmation[requester_task_id]

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
        # Process rest of time step
        # Add debug prints for tasks creation
        created_tasks = self.tasks_autocreation()
        if created_tasks:
            print(f"\n🔍 Created {len(created_tasks)} tasks:")
            for task in created_tasks:
                print(f"Task {task['task_id']}: {task['task_name']} from {task['starting_sensor']} to controller {task['target']}")

        self.process_confirmations()
        self.process_tasks()
        
        for connection in self.graph.get("connections", []):
            # Calculate available bandwidth for this connection
            available_bandwidth = ((connection["speed"] * 1000) / 1000) * self.time_step_in_ms
            available_bandwidth += connection["reserved_from_previous_step"]
            
            connection["reserved_from_previous_step"] = 0
            
            if not connection["tasks"]:
                continue
            
            # Sort tasks by importance to ensure high priority tasks are processed first
            connection["tasks"].sort(key=lambda x: x.get("importance", 0), reverse=True)
            
            tasks_to_remove = []
            for task in connection["tasks"]:
                if available_bandwidth >= task["task_size"]:
                    target = int(connection["target"]) + int(connection["source"]) - int(task["task_location"])
                    if self.do_task(task, target):
                        tasks_to_remove.append(task)
                        available_bandwidth -= task["task_size"]
                else:
                    connection["reserved_from_previous_step"] = available_bandwidth
                    break
            
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
    env = GraphEnv(usfl_arr.descriptions_for_regular_tasks)
    obs, info = env.reset()
    
    print("\n=== Test: Light Control Without Confirmations ===")
    # Create light control tasks from controller 4 to multiple lights
    light_tasks, success = env.lights_scenario("4", importance=5)
    
    if success:
        print(f"Successfully created {len(light_tasks)} light control tasks")
        
        # Run until all tasks are complete
        obs, reward, done, info = env.step([])
        
        # Print final results
        print("\n=== Final Results ===")
        print(f"Steps taken: {info['steps_taken']}")
        print(f"Total time elapsed: {info['time']}ms")
        print(f"Total reward accumulated: {reward}")
        
        # Check final state of lights
        for light_id in usfl_arr.light:
            light = env.find_edge_vertex(light_id)
            if light:
                print(f"\nLight {light_id} final state:")
                print(f"IsOn: {light.get('isOn', 'N/A')}")
                print(f"Brightness: {light.get('brightness', 'N/A')}")
    else:
        print("Failed to create light control tasks")
