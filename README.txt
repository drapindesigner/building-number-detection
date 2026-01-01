
Machine Perception Assignment

Directory Structure:
---------------------
Your submission folder should be named using the convention [lastname]_[firstname]_[studentID] 
Example: trump_donald_12345678
Please remove spaces, dashes and other non-alphabet characters in your lastname and firstname whilst creating this folder.

Inside this folder, you must have the following structure:

- output/ 
    - task1/ : Directory for Task 1 output files.
    - task2/ : Directory for Task 2 output files.
    - task3/ : Directory for Task 3 output files.
    - task4/ : Directory for Task 4 output files.
    
- packages/ : Folder containing any Python packages that are not installable via pip (optional).
- data/ : Folder containing any pre-trained weights/checkpoints required for your models (if any).

Files:
------
- assignment.py : **Do not modify this file**. It handles execution for tasks 1, 2, 3, and 4. 
  Focus on completing the task files (task1.py, task2.py, task3.py, task4.py) to implement your solution.
  
  Example usage:
  `python assignment.py task1 /path/to/images/for/task1`
  `python assignment.py task2 /path/to/images/for/task2`

- task1.py, task2.py, task3.py, task4.py : Complete these files to implement the functionality for each task. Each task should save the output to the designated output folder.

- requirements.txt : List of acceptable Python libraries for your project.

What to Modify:
---------------
1. Implement the methods in task1.py, task2.py, task3.py, and task4.py. 
2. Ensure your output is written to the correct subfolders under the output/ directory.
3. **Do not modify assignment.py**. It is already configured to handle the tasks and pass arguments correctly.