import random
CO2 = [
    112, 133, 154, 174, 195, 216, 237, 258, 279, 300, 322, 343, 364, 385, 406, 427,
    465, 467, 469, 471, 473, 475, 477, 479, 481, 483, 485, 487, 489, 491, 493, 495,
    497, 638
]

LiDAR = [
    98, 131, 152, 172, 193, 214, 235, 256, 277, 298, 320, 341, 362, 383, 404, 425, 636
]

controllers = [3, 4, 7, 10]

door = [
    111, 132, 153, 173, 194, 215, 236, 257, 278, 299, 321, 342, 363, 384, 405, 426, 637
]

light = [
    76, 78, 80, 81, 83, 85, 87, 89, 95, 96, 97, 100, 102, 104, 107, 117, 119, 121,
    122, 124, 126, 128, 130, 138, 140, 142, 143, 145, 147, 149, 151, 158, 160, 162,
    163, 165, 167, 169, 171, 179, 181, 183, 184, 186, 188, 190, 192, 200, 202, 204,
    205, 207, 209, 211, 213, 221, 223, 225, 226, 228, 230, 232, 234, 242, 244, 246,
    247, 249, 251, 253, 255, 263, 265, 267, 268, 270, 272, 274, 276, 284, 286, 288,
    289, 291, 293, 295, 297, 306, 308, 310, 311, 313, 315, 317, 319, 327, 329, 331,
    332, 334, 336, 338, 340, 348, 350, 352, 353, 355, 357, 359, 361, 369, 371, 373,
    374, 376, 378, 380, 382, 390, 392, 394, 395, 397, 399, 401, 403, 411, 413, 415,
    416, 418, 420, 422, 424, 622, 624, 626, 627, 629, 631, 633, 635
]

movement = [29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59]

noise = [11, 13, 16, 18, 20, 22, 25, 27]

temperature = [
    114, 135, 155, 176, 197, 218, 239, 260, 281, 302, 324, 345, 366, 387, 408, 429, 640
]

ventelation = [
    113, 134, 175, 196, 217, 238, 259, 280, 301, 323, 344, 365, 386, 407, 428, 517,
    534, 537, 539, 541, 543, 545, 547, 549, 551, 553, 555, 556, 558, 560, 562, 564,
    566, 568, 570, 572, 574, 576, 578, 580, 582, 584, 586, 588, 590, 592, 594, 596,
    598, 600, 602, 605, 607, 609, 611, 613, 616, 617, 639
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
        "importance": 7,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "noise",
        "frequency": 2000,
        "importance": 8,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "movement",
        "frequency": 2000,
        "importance": 9,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "light",
        "frequency": 2000,
        "importance": 5,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "LiDAR",
        "frequency": 1000,
        "importance": 7,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    },
    {
        "label": "CO2",
        "frequency": 500,
        "importance": 9,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [],
        "target": None,
    }
]

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
    new_task = {
        "label": label,
        "frequency": 1000,
        "importance": 5,
        "task_size": 150,
        "sram_usage": 3000,
        "specific_ids": [specific_id],
        "target": target
    }
    descriptions_for_regular_tasks.append(new_task)

