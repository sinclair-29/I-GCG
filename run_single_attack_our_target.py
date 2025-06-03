import time
import random
import datetime
import logging # logging 模块本身可能仍然有用，即使不用多线程
from run_single_attack_base import run_single_process
import os
# make the timestamp utc-8
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--defense', type=str, default="no_defense")
parser.add_argument('--behaviors_config', type=str, default="behaviors_config.json")
parser.add_argument('--output_path', type=str,
                    default='ours')
# 新增参数，允许用户指定GPU ID
parser.add_argument('--gpu_id', type=int, default=0, help="The ID of the single GPU to use")


args = parser.parse_args()

# --- 修改：使用单个GPU ---
# device_list = [0,1,2,3] # 原来的多GPU列表
single_gpu_id = args.gpu_id # 使用命令行指定的GPU ID，默认为0
# --- 修改结束 ---

defense=args.defense
timestamp = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")

output_path=os.path.join("Our_GCG_target_len_20",args.output_path)
output_path=os.path.join(output_path,str(timestamp))

# 确保输出目录存在
if not os.path.exists(output_path):
    os.makedirs(output_path)
    print(f"Created output directory: {output_path}")


behaviors_config=args.behaviors_config
behavior_id_list = [i + 1 for i in range(50)]

tasks_to_process = list(behavior_id_list) # 创建一个副本以防原始列表被意外修改

print(f"Starting processing on single GPU: {single_gpu_id}")
print(f"Output path: {output_path}")
print(f"Defense: {defense}")
print(f"Behaviors config: {behaviors_config}")
print(f"Total tasks to process: {len(tasks_to_process)}")

for i, task_id in enumerate(tasks_to_process):
    print(f"--- Processing task {i+1}/{len(tasks_to_process)} (ID: {task_id}) using GPU {single_gpu_id} ---")
    start_time = time.time()
    try:
        run_single_process(task_id, single_gpu_id, output_path, defense, behaviors_config)
        end_time = time.time()
        print(f"--- Successfully completed task {task_id}. Time taken: {end_time - start_time:.2f} seconds ---")
    except Exception as e:
        end_time = time.time()
        print(f"!!! Error processing task {task_id}: {e}. Time taken: {end_time - start_time:.2f} seconds !!!")
        logging.error(f"Error processing task {task_id} on GPU {single_gpu_id}: {e}", exc_info=True)


print("All tasks completed!")