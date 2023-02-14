import numpy as np
from fractions import Fraction
from collections import namedtuple
from ast import literal_eval as make_tuple

import re
import os


def extract(string):
    """
    Extracts string as decimal or int
    """
    if "." in string:
        return Fraction(string)
    else:
        return int(string)


TaskID = namedtuple("TaskID", ["task_space", "task_id"])

TaskInfo = namedtuple("TaskInfo", [
    "task_id", "task_runtime", "task_dependencies", "data_dependencies"])

TaskRuntimeInfo = namedtuple("TaskRuntimeInfo", [
    "task_time", "device_fraction", "gil_accesses", "gil_fraction", "memory"])

TaskDataInfo = namedtuple("TaskDataInfo", ['read', 'write', 'read_write'])


def read_gphx(filename):
    """
    Reads a graphx file and returns:
    1. A list of the nodes in the graph
    2. The initial data configuration
    """

    task_list = []
    data_config = dict()

    with open(filename, 'r') as graph_file:

        lines = graph_file.readlines()

        # Read the initial data configuration
        data_info = lines.pop(0)
        data_info = data_info.split(',')
        # print(data_info)
        idx = 0
        for data in data_info:
            info = data.strip().strip("{}").strip().split(":")
            size = int(info[0].strip())
            location = int(info[1].strip())
            data_config[idx] = (size, location)
            idx += 1

        # print("Data Config", data_config)
        # Read the task graph
        for line in lines:

            task = line.split("|")
            # Breaks into [task_id, task_runtime, task_dependencies, data_dependencies]

            # Process task id (can't be empty)
            ids = task[0].strip()
            ids = make_tuple(ids)

            if not isinstance(ids, tuple):
                ids = (ids,)

            if isinstance(ids[0], str) and ids[0].isalpha():
                task_space = ids[0]
                idx = ids[1]

                if not isinstance(idx, tuple):
                    idx = (idx, )

                task_ids = TaskID(task_space, idx)
            else:
                task_space = "T"
                task_ids = TaskID(task_space, ids)

            # Process task runtime (can't be empty)
            configurations = task[1].strip().split("},")
            task_runtime = dict()
            for config in configurations:
                config = config.strip().strip("{}").strip()
                config = config.split(":")

                targets = config[0].strip().strip("()").strip().split(",")
                targets = [int(target.strip()) for target in targets]
                target = tuple(targets)

                details = config[1].strip().split(",")
                details = [extract(detail.strip()) for detail in details]
                details = TaskRuntimeInfo(*details)

                task_runtime[target] = details

            # Process task dependencies (can be empty)
            if len(task) > 2:
                dependencies = task[2].split(":")
                if (len(dependencies) > 0) and (not dependencies[0].isspace()):
                    task_dependencies = []

                    for i in range(len(dependencies)):
                        if not dependencies[i].isspace():
                            ids = dependencies[i].strip()

                            ids = make_tuple(ids)

                            if not isinstance(ids, tuple):
                                ids = (ids,)

                            if isinstance(ids[0], str) and ids[0].isalpha():
                                name, idx = ids[0], ids[1]

                                if not isinstance(idx, tuple):
                                    idx = (idx, )
                                dep_id = TaskID(name, idx)

                            else:
                                dep_id = TaskID(task_space, ids)

                            task_dependencies.append(dep_id)
                else:
                    task_dependencies = []

            else:
                task_dependencies = []

            task_dependencies = task_dependencies

            # Process data dependencies (can be empty)
            if len(task) > 3:
                # Split into [read, write, read/write]
                types = task[3].split(":")

                check = [not t.isspace() for t in types]

                if any(check):
                    task_data = [[], [], []]

                    for i in range(len(types)):
                        if check[i]:
                            data = types[i].strip().split(",")
                            if not data[0].isspace():
                                task_data[i] = [0 for _ in range(len(data))]

                                for j in range(len(data)):
                                    if not data[j].isspace():
                                        task_data[i][j] = int(data[j])
                else:
                    task_data = [[], [], []]
            else:
                task_data = [[], [], []]

            task_data = TaskDataInfo(*task_data)

            task_tuple = TaskInfo(task_ids, task_runtime,
                                  task_dependencies, task_data)

            task_list.append(task_tuple)

    return data_config, task_list


def convert_to_dictionary(task_list):
    """
    Converts a task list to a dictionary
    """
    task_dict = dict()
    for task in task_list:
        task_dict[task.task_id] = task

    return task_dict


START_TASK = 1
COMPLETED_TASK = 2

task_filter = re.compile(r'InnerTask\{ .*? \}')


TaskTime = namedtuple('TaskTime', ['start_time', 'end_time', 'duration'])


def get_time(line):
    time = line.split('>>')[0].strip().strip("\`").strip('[]')
    return int(time)


def check_line(line):
    if "Running task" in line:
        return START_TASK
    elif "Completed task" in line:
        return COMPLETED_TASK


def convert_task_id(task_id):
    id = task_id.strip().split('_')
    task_space = id[0]
    task_idx = tuple([int(i) for i in id[1:]])
    return TaskID(task_space, task_idx)


def get_task_properties(line):
    message = line.split('>>')[1].strip()
    tasks = re.findall(task_filter, message)
    for task in tasks:
        properties = {}
        task = task.strip('InnerTask{').strip('}').strip()
        task_properties = task.split(',')
        for prop in task_properties:
            prop_name, prop_value = prop.strip().split(':')
            properties[prop_name] = prop_value.strip()

        properties['name'] = convert_task_id(properties['name'])

    return properties


def parse_blog():
    filename = 'parla.blog'

    result = subprocess.run(
        ['bread', '-s', r"-f `[%r] >> %m`", filename], stdout=subprocess.PIPE)

    output = result.stdout.decode('utf-8')
    output = output.splitlines()

    task_start_times = {}
    task_end_times = {}

    task_start_order = []
    task_end_order = []

    task_times = {}

    for line in output:
        line_type = check_line(line)
        if line_type == START_TASK:
            start_time = get_time(line)
            task_properties = get_task_properties(line)
            task_start_times[task_properties['name']] = start_time
            task_start_order.append(task_properties['name'])

        elif line_type == COMPLETED_TASK:
            print("Completed task")
            end_time = get_time(line)
            task_properties = get_task_properties(line)
            print(end_time - start_time)
            task_end_times[task_properties['name']] = end_time
            task_end_order.append(task_properties['name'])

    for task in task_end_times:
        start_t = task_start_times[task]
        end_t = task_end_times[task]
        duration = end_t - start_t
        task_times[task] = TaskTime(start_t, end_t, duration)

    return task_times, task_start_order, task_end_order
