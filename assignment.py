# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Author: Mohammad Saif Ul Haq
# Last Modified: 2025-10-03

import sys
import os

# Importing the individual tasks
from task1 import run_task1
from task2 import run_task2
from task3 import run_task3
from task4 import run_task4

# Function to read the config file
def read_config(config_path):
    config = {}
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return config

    with open(config_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('#'):
                key, value = line.split(':')
                config[key.strip()] = value.strip()
    return config

def print_usage():
    print("Usage: assignment.py <task> <image_path> [config_path]")
    print("Example:")
    print("  python assignment.py task1 /mnt/data/comp3007/assignment2024/task1")
    print("  Optional config file path (default: config.txt) in the same directory as assignment.py")
    print("  python assignment.py task1 /mnt/data/comp3007/assignment2024/task1 ./config.txt")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Error: Incorrect number of arguments.")
        print_usage()
        sys.exit(1)

    task = sys.argv[1]
    image_path = sys.argv[2]
    config_path = sys.argv[3] if len(sys.argv) > 3 else 'config.txt'

    config = read_config(config_path)
    if not config:
        sys.exit(1)

    try:
        if task == "task1":
            run_task1(image_path, config)
        elif task == "task2":
            run_task2(image_path, config)
        elif task == "task3":
            run_task3(image_path, config)
        elif task == "task4":
            run_task4(image_path, config)
        else:
            print(f"Unknown task: {task}. Please specify task1, task2, task3, or task4.")
            print_usage()
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred while executing {task}: {str(e)}")
        sys.exit(1)