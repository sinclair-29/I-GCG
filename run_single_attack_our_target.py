# import threading # Removed
# import time # Removed, unless run_single_process needs it for other reasons
# import random # Removed, unless run_single_process needs it for other reasons
import datetime
# import logging # Was not used in the original snippet, can be removed if not used by run_single_process
from run_single_attack_base import run_single_process
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--defense', type=str, default="no_defense")
parser.add_argument('--behaviors_config', type=str, default="behaviors_config.json")
parser.add_argument('--output_path', type=str,
                    default='ours')
# Add an argument for specifying the single GPU device ID
parser.add_argument('--device_id', type=int, default=0, help="ID of the single GPU to use (e.g., 0 for A100)")


args = parser.parse_args()

# Use the specified single device ID
single_device_id = args.device_id

defense = args.defense
timestamp = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

output_path = os.path.join("Our_GCG_target_len_20", args.output_path)
output_path = os.path.join(output_path, str(timestamp))

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created output directory: {output_path}")

behaviors_config_file = args.behaviors_config # Renamed for clarity
behavior_id_list = [i + 1 for i in range(50)] # Assuming 50 behaviors

# Optional: Add id to black_list to skip the id
# black_list = [10, 25]
# behavior_id_list = [i for i in behavior_id_list if i not in black_list]

# Optional: Add id to white_list to only run the id
# white_list = [1, 5, 7]
# behavior_id_list = [i for i in behavior_id_list if i in white_list]


print(f"Starting sequential processing on device GPU:{single_device_id}")
print(f"Total tasks to process: {len(behavior_id_list)}")
print(f"Output will be saved to: {output_path}")
print(f"Behaviors config file: {behaviors_config_file}")
print(f"Defense mode: {defense}")

# Sequentially process each task
for task_id in behavior_id_list:
    print(f"Processing task {task_id} using device GPU:{single_device_id}...")
    try:
        # The run_single_process function is called directly.
        # It needs the task_id, the device_id, output_path, defense, and behaviors_config file.
        run_single_process(task_id, single_device_id, output_path, defense, behaviors_config_file)
        print(f"Completed task {task_id}.")
    except Exception as e:
        print(f"Error processing task {task_id}: {e}")
        # Optionally, you might want to log this error to a file or continue to the next task.

print("All tasks completed!")