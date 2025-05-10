import random
import copy

CO2 = [
    44, 56, 68, 79, 91, 103, 115, 127, 139, 151,
163, 175, 187, 199, 211, 223, 227, 228, 229, 230,
231, 232, 233, 234, 235, 236, 237, 238, 239, 240,
241, 242, 243, 296
]

LiDAR = [
    39, 54, 66, 77, 89, 101, 113, 125, 137, 149, 161, 173, 185, 197, 209, 221, 294
]

controllers = [1,2,3,4]

door = [
    43, 55, 67, 78, 90, 102, 114, 126, 138, 150, 162, 174, 186, 198, 210, 222, 295
]

light = [
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38 ,40, 41, 42, 47, 48, 49, 50, 51, 52, 53, 59, 60,
    61, 62, 63, 64, 65, 70, 71, 72, 73, 74, 75, 76, 82, 83, 84, 85, 86, 87, 88, 94, 95, 96,
    97, 98, 99, 100, 106, 107, 108, 109, 110, 111, 112, 118, 119, 120, 121, 122, 123, 124,
    130, 131, 132, 133, 134, 135, 136, 142, 143, 144, 145, 146, 147, 148, 154, 155, 156, 157,
    158, 159, 160, 166, 167, 168, 169, 170, 171, 172, 178, 179, 180, 181, 182, 183, 184, 190,
    191, 192, 193, 194, 195, 196, 202, 203, 204, 205, 206, 207, 208, 214, 215, 216, 217, 218,
    219, 220, 287, 288, 289, 290, 291, 292, 293, 299, 300, 301, 302, 303, 304, 305, 306, 307,
    308, 309, 310, 311, 312, 313, 314, 315, 316
]

movement = [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

noise = [5, 6, 7, 8, 9, 10, 11, 12]

temperature = [
    46, 58, 69, 81, 93, 105, 117, 129, 141, 153,165, 177, 189, 201, 213, 225, 298
]

ventelation = [
    45, 57, 80, 92, 104, 116, 128, 140, 152, 164, 176, 188, 200, 212, 224, 244, 245, 246,
    247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
    264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280,
    281, 282, 283, 284, 285, 286, 297
]
sensors = CO2 + LiDAR + door + light + movement + noise + temperature + ventelation
rooms = {
 'room10': {'CO2': 495,
            'LiDAR': 193,
            'controller': 7,
            'door': 173,
            'lights': [158, 160, 162, 163, 165, 167, 169, 171],
            'temperature': 155,
            'ventelation': 196},
 'room11': {'CO2': 195,
            'LiDAR': 636,
            'controller': 7,
            'door': 153,
            'lights': [179, 181, 183, 184, 186, 188, 190, 192],
            'temperature': 176,
            'ventelation': 175},
 'room12': {'CO2': 174,
            'LiDAR': None,
            'controller': 7,
            'door': 637,
            'lights': [622, 624, 626, 627, 629, 631, 633, 635],
            'temperature': 197,
            'ventelation': 590},
 'room13': {'CO2': 322,
            'LiDAR': 425,
            'controller': 10,
            'door': 405,
            'lights': [306, 308, 310, 311, 313, 315, 317, 319],
            'temperature': 366,
            'ventelation': 386},
 'room14': {'CO2': 471,
            'LiDAR': 404,
            'controller': 10,
            'door': 426,
            'lights': [327, 329, 331, 332, 334, 336, 338, 340],
            'temperature': 387,
            'ventelation': 365},
 'room15': {'CO2': 475,
            'LiDAR': 341,
            'controller': 10,
            'door': 342,
            'lights': [348, 350, 352, 353, 355, 357, 359, 361],
            'temperature': 345,
            'ventelation': 344},
 'room16': {'CO2': 497,
            'LiDAR': 362,
            'controller': 10,
            'door': 363,
            'lights': [369, 371, 373, 374, 376, 378, 380, 382],
            'temperature': 408,
            'ventelation': 407},
 'room17': {'CO2': 427,
            'LiDAR': 383,
            'controller': 10,
            'door': 384,
            'lights': [390, 392, 394, 395, 397, 399, 401, 403],
            'temperature': 429,
            'ventelation': 428},
 'room18': {'CO2': 406,
            'LiDAR': 320,
            'controller': 10,
            'door': 321,
            'lights': [411, 413, 415, 416, 418, 420, 422, 424],
            'temperature': 324,
            'ventelation': 323},
 'room2': {'CO2': 237,
           'LiDAR': 98,
           'controller': 4,
           'door': 299,
           'lights': [95, 96, 97, 100, 102, 104, 107],
           'temperature': 281,
           'ventelation': 113},
 'room3': {'CO2': 279,
           'LiDAR': 277,
           'controller': 4,
           'door': 215,
           'lights': [200, 202, 204, 205, 207, 209, 211, 213],
           'temperature': 302,
           'ventelation': 280},
 'room4': {'CO2': 469,
           'LiDAR': 256,
           'controller': 4,
           'door': 257,
           'lights': [221, 223, 225, 226, 228, 230, 232, 234],
           'temperature': 114,
           'ventelation': 301},
 'room5': {'CO2': 481,
           'LiDAR': 298,
           'controller': 4,
           'door': 236,
           'lights': [242, 244, 246, 247, 249, 251, 253, 255],
           'temperature': 218,
           'ventelation': 217},
 'room6': {'CO2': 493,
           'LiDAR': 214,
           'controller': 4,
           'door': 111,
           'lights': [263, 265, 267, 268, 270, 272, 274, 276],
           'temperature': 260,
           'ventelation': 238},
 'room7': {'CO2': 300,
           'LiDAR': 235,
           'controller': 4,
           'door': 278,
           'lights': [284, 286, 288, 289, 291, 293, 295, 297],
           'temperature': 239,
           'ventelation': 259},
 'room8': {'CO2': 487,
           'LiDAR': 152,
           'controller': 7,
           'door': 132,
           'lights': [117, 119, 121, 122, 124, 126, 128, 130],
           'temperature': 135,
           'ventelation': 517},
 'room9': {'CO2': 491,
           'LiDAR': 131,
           'controller': 7,
           'door': 194,
           'lights': [138, 140, 142, 143, 145, 147, 149, 151],
           'temperature': 640,
          'ventelation': 134}}

descriptions_for_regular_tasks = [
    {
        "label": "temperature",
        "frequency": 500,
        "importance": 4,
        "task_size": 50,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "noise",
        "frequency": 2000,
        "importance": 4,
        "task_size": 50,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "movement",
        "frequency": 2000,
        "importance": 6,
        "task_size": 50,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "light",
        "frequency": 2000,
        "importance": 2,
        "task_size": 50,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "LiDAR",
        "frequency": 1000,
        "importance": 6,
        "task_size": 50,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "CO2",
        "frequency": 500,
        "importance": 7,
        "task_size": 50,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    }
]

# Store the initial state
_initial_descriptions_for_regular_tasks = copy.deepcopy(descriptions_for_regular_tasks)

def add_new_regular_task(target, specific_id,label):
    label = str(label)
    """
    Adds a new task to descriptions_for_regular_tasks list or updates existing task
    if same label and target exists
    Args:
        target (int): Controller ID
        specific_id (int): Sensor ID
    """
    # Check if task with same label and target exists
    for task in descriptions_for_regular_tasks:
        if task["label"] == label and task["target"] == target:
            # Add specific_id to existing task if not already present
            if specific_id not in task["specific_ids"]:
                task["specific_ids"].append(specific_id)
            return

    # If no matching task found, create new one
    # Find the default task properties based on the label
    default_task_props = next((item for item in _initial_descriptions_for_regular_tasks if item["label"] == label), None)

    if default_task_props is None:
        # Fallback or error handling if label not found, though prompt says assume it exists
        # For now, let's use some default values or raise an error
        # Using the provided example's fallback for now:
        new_task = {
            "label": label,
            "frequency": 1000,  # Default if label not found
            "importance": 5,    # Default if label not found
            "task_size": 50,    # Default if label not found
            "sram_usage": 3000, # Default if label not found
            "specific_ids": [specific_id],
            "target": target
        }
    else:
        new_task = {
            "label": label,
            "frequency": default_task_props["frequency"],
            "importance": default_task_props["importance"],
            "task_size": default_task_props["task_size"],
            "sram_usage": default_task_props["sram_usage"],
            "specific_ids": [specific_id],
            "target": target
        }
    descriptions_for_regular_tasks.append(new_task)

def reset_regular_tasks():
    """Resets the descriptions_for_regular_tasks list to its initial state."""
    global descriptions_for_regular_tasks
    descriptions_for_regular_tasks = copy.deepcopy(_initial_descriptions_for_regular_tasks)

print (len(CO2) + len(LiDAR) + len(movement) + len(noise))