from parla.utility.graphs import generate_single_device_serial, SerialConfig
from parla.utility.graphs import TaskConfig, TaskConfigs, DataGraphConfig, IndependentConfig, Device, Architecture
from parla.utility.graphs import RunConfig, TaskDataInfo, DataInfo, TaskInfo, TaskID, TaskRuntimeInfo
from typing import Tuple, Dict
from ast import literal_eval as make_tuple
from rich import print

import yaml 
from fractions import Fraction
from decimal import Decimal


def write_object_to_dict(obj, ):
    """
    Write a single task to an open YAML file
    """
    sub_dict = {}

    def is_base(x): return isinstance(
        x, (int, float, str, bool, type(None), Fraction, Decimal))

    def is_base_str(x): return isinstance(x, (tuple, Architecture, Device))

    if isinstance(obj, Dict):
        for key, value in obj.items():
            key = str(key)
            if is_base_str(value):
                sub_dict[key] = str(value)
            elif is_base(value):
                sub_dict[key] = value
            elif isinstance(value, list):
                sub_dict[key] = [write_object_to_dict(x) for x in value]
            else:
                sub_dict[key] = write_object_to_dict(value)
    elif is_base(obj):
        return obj
    else:
        for slot in obj.__slots__:
            value = getattr(obj, slot)
            if is_base_str(value):
                sub_dict[slot] = str(value)
            elif is_base(value):
                sub_dict[slot] = value
            elif isinstance(value, list):
                sub_dict[slot] = [write_object_to_dict(x) for x in value]
            else:
                sub_dict[slot] = write_object_to_dict(value)

    return sub_dict

def write_to_pgraph(tasks: Dict[TaskID, TaskInfo], data: Dict[int, DataInfo], basename: str = "graph"):
    """
    Write the task graph to a pgraph file
    """

    taskfile = basename + ".pgraph"

    def info_to_comma(info):
        comma = ", ".join(
            [f"{getattr(info, slot)}" for slot in info.__slots__])
        return comma
    
    def unpack_runtime(runtime: TaskRuntimeInfo | Dict[Device | Tuple[Device], TaskRuntimeInfo]):
        if isinstance(runtime, Dict):
            return ", ".join([f"{{{device} : {unpack_runtime(r)}}}" for device,
                    r in runtime.items()])
        elif isinstance(runtime, TaskRuntimeInfo):
            return info_to_comma(runtime)
        else:
            raise ValueError(f"Unknown runtime type {runtime}")

    def unpack_id(task_id):
        return f"('{task_id.taskspace}', {tuple(task_id.task_idx)})"
    
    def unpack_dependencies(dependencies: list[TaskID]):
        return ": ".join( [f" {unpack_id(task_id)} " for task_id in dependencies] )

    def unpack_data_dependencies(dependencies: TaskDataInfo):
        read_data = dependencies.read
        write_data = dependencies.write
        read_write_data = dependencies.read_write

        read_string = ", ".join([f"{data}" for data in read_data])
        write_string = ", ".join([f"{data}" for data in write_data])
        read_write_string = ", ".join([f"{data}" for data in read_write_data])

        return f"{read_string} : {write_string} : {read_write_string}"


    data_line = ", ".join([f"{{{data.size} : {data.location}}}" for data in data.values()])
    task_lines = []


    for task in tasks.values():
        task_id = unpack_id(task.task_id)
        task_runtime = unpack_runtime(task.task_runtime)
        task_dependencies = unpack_dependencies(task.task_dependencies)
        task_data_dependencies = unpack_data_dependencies(task.data_dependencies)
        
        task_line = f"{task_id} | {task_runtime} | {task_dependencies} | {task_data_dependencies}"
        task_lines.append(task_line)

    with open(taskfile, "w") as file:
        print(data_line, file=file)
        for task_line in task_lines:
            print(task_line, file=file)


def extract(string: str) -> int | Fraction:
    """
    Extracts string as decimal or int
    """
    if "." in string:
        return Fraction(string)
    else:
        return int(string)


def read_from_pgraph(basename: str = "graph") -> Tuple[Dict[TaskID, TaskInfo], Dict[int, DataInfo]]:

    task_dict = dict()
    data_dict = dict()

    filename = basename + ".pgraph"

    def extract_task_id(line: str) -> TaskID:
        ids = line.strip()
        ids = make_tuple(ids)

        if not isinstance(ids, tuple):
            ids = (ids,)

        if isinstance(ids[0], str) and ids[0].isalpha():
            taskspace = ids[0]
            task_idx = ids[1]

            if not isinstance(task_idx, tuple):
                task_idx = (task_idx,)
        
        else:
            taskspace = "T"
            task_idx = ids 
        
        return TaskID(taskspace, task_idx, 0)

    def extract_task_runtime(line: str) -> Dict[Device | Tuple[Device], TaskRuntimeInfo]:

        line = line.strip()
        
        configurations = line.split("},")
        configurations = [config.strip().strip("{}").strip() for config in configurations]
        task_runtime = {}
        for config in configurations:
            targets, details = config.split(":")
            targets = device_from_string(targets)

            details = [extract(detail.strip()) for detail in details.split(",")]

            task_runtime[targets] = TaskRuntimeInfo(*details)

        return task_runtime
    
    def extract_task_dependencies(line: str) -> list[TaskID]:
        line = line.strip()
        dependencies = line.split(":")
        dependencies = [dependency.strip() for dependency in dependencies]

        if dependencies[0] == "":
            return []

        return [extract_task_id(dependency) for dependency in dependencies]


    def extract_data_dependencies(line: str) -> TaskDataInfo:
        line = line.strip()
        dependencies = line.split(":")
        dependencies = [dependency.strip() for dependency in dependencies]

        check_has = [(not dependency.isspace()) and (not dependency == '')
                     for dependency in dependencies]

        if not any(check_has):
            return TaskDataInfo([], [], [])

        if len(dependencies) > 3:
            raise ValueError(f"Too many data movement types {dependencies}")
        
        if len(dependencies) < 1 or dependencies[0].isspace() or not check_has[0]:
            read_data = []
        else:
            read_data = [int(x.strip()) for x in dependencies[0].split(",")]

        if len(dependencies) < 2 or dependencies[1].isspace() or not check_has[1]:
            write_data = []
        else:
            write_data = [int(x.strip()) for x in dependencies[1].split(",")]

        if len(dependencies) < 3 or dependencies[2].isspace() or not check_has[2]:
            read_write_data = []
        else:
            read_write_data = [int(x.strip()) if (x) else None for x in dependencies[2].split(",")]

        return TaskDataInfo(read_data, write_data, read_write_data)

    with open(filename, "r") as file:

        lines = file.readlines()

        data_line = lines.pop(0)
        data_line = data_line.strip()
        data_line = data_line.split(",")

        for idx, data in enumerate(data_line):
            data = data.strip()
            data = data.strip("{}")
            data = data.split(":")
            data_size = int(data[0])
            data_location = device_from_string(data[1])
            data_info = DataInfo(idx, data_size, data_location)
            data_dict[idx] = data_info

        for task_line in lines:
            task = task_line.strip()
            task = task.split("|")
            
            if len(task) < 2:
                raise ValueError(f"Task line {task_line} is too short")
            
            task_id = extract_task_id(task[0])
            task_runtime = extract_task_runtime(task[1])
            if len(task) > 2:
                task_dependencies = extract_task_dependencies(task[2])
            else:
                task_dependencies = []

            if len(task) > 3:
                task_data_dependencies = extract_data_dependencies(task[3])
            else:
                task_data_dependencies = TaskDataInfo([], [], [])

            task_info = TaskInfo(task_id, task_runtime, task_dependencies, task_data_dependencies)
            task_dict[task_id] = task_info

    return task_dict, data_dict
            
            




        






def write_to_yaml(tasks: Dict[TaskID, TaskInfo], data: Dict[int, DataInfo], basename: str = "graph"):
    """
    Write the task graph to a yaml file
    """

    taskfile = basename + ".tasks.yaml"
    datafile = basename + ".data.yaml"

    with open(taskfile, "w") as file:
        tasks = [write_object_to_dict(task) for task in tasks.values()]
        yaml.dump(tasks, file, default_flow_style=False, sort_keys=False)

    with open(datafile, "w") as file:
        data = [write_object_to_dict(data) for data in data.values()]
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def read_from_yaml(basename: str = "graph") -> Tuple[Dict[TaskID, TaskInfo], Dict[int, DataInfo]]:
    """
    Read the task graph from a yaml file
    """

    taskfile = basename + ".tasks.yaml"
    datafile = basename + ".data.yaml"

    task_dict = dict()
    with open(taskfile, "r") as file:
        tasks = yaml.load(file, Loader=yaml.FullLoader)
        tasks = [read_tasks_from_dict(task) for task in tasks]
    
    for task in tasks:
        task_dict[task.task_id] = task

    data_dict = dict()
    with open(datafile, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        data = [make_data_info(data) for data in data]

    for data in data:
        data_dict[data.idx] = data

    return task_dict, data_dict


def make_data_info(data_info: Dict) -> DataInfo:
    data_idx = int(data_info["idx"])
    data_size = int(data_info["size"])
    data_location = device_from_string(data_info["location"])

    return DataInfo(data_idx, data_size, data_location)


def make_task_id(task_id: Dict) -> TaskID:
    taskspace = task_id["taskspace"]
    task_idx = make_tuple(task_id["task_idx"])
    task_instance = int(task_id["instance"])
    return TaskID(taskspace, task_idx, task_instance)


def make_data_dependencies(data_dependencies: Dict) -> TaskDataInfo:
    read_data = [int(x) for x in data_dependencies["read"]]
    write_data = [int(x) for x in data_dependencies["write"]]
    read_write_data = [int(x) for x in data_dependencies["read_write"]]
    return TaskDataInfo(read_data, write_data, read_write_data)

def make_task_runtime(task_runtime: Dict) -> TaskRuntimeInfo:
    task_time = int(task_runtime["task_time"])
    device_fraction = Fraction(task_runtime["device_fraction"])
    gil_accesses = int(task_runtime["gil_accesses"])
    gil_fraction = Fraction(task_runtime["gil_fraction"])
    memory = int(task_runtime["memory"])

    return TaskRuntimeInfo(task_time, device_fraction, gil_accesses, gil_fraction, memory)


def device_from_string(device_str: str) -> Device | Tuple[Device]:
    """
    Convert a device string (or string of a device tuple) to a device set
    """
    if device_str is None:
        return None
    device_str = device_str.strip()
    device_str = device_str.strip("()")
    device_str = device_str.strip()
    device_str = device_str.split(",")
    device_str = [d.strip() for d in device_str]

    devices = []

    for d in device_str:
        d = d.strip()
        d = d.strip("]")
        d = d.split("[")

        if d[0] == "CPU":
            devices.append(Device(Architecture.CPU, int(d[1])))
        elif d[0] == "GPU":
            devices.append(Device(Architecture.GPU, int(d[1])))
        elif d[0] == "ANY":
            devices.append(Device(Architecture.ANY, int(d[1])))
        else:
            raise ValueError(f"Unknown device type {d[0]}")

    if len(devices) == 1:
        return devices[0]
    else:
        return tuple(devices)

def parse_device_runtime(task_runtime: Dict) -> Dict[Device | Tuple[Device], TaskRuntimeInfo | Dict[Device | Tuple[Device], TaskRuntimeInfo]]:
    """
    Parse the device runtime from a dictionary
    """
    device_runtime = {}
    for device_str, runtime in task_runtime.items():
        device = device_from_string(device_str)

        if 'task_time' in runtime:
            device_runtime[device] = make_task_runtime(runtime)
        elif isinstance(device, tuple):
            device_runtime[device] = parse_device_runtime(runtime)
        else:
            raise ValueError(f"Unknown device type {device} or Invalid runtime {runtime} configuration.")

    return device_runtime


def read_tasks_from_dict(task_dict: Dict) -> TaskInfo:
    """
    Read a task from a dictionary
    """
     
    task_id = make_task_id(task_dict["task_id"])
    

    task_runtime = parse_device_runtime(task_dict["task_runtime"])
    task_dependencies = [make_task_id(task) for task in task_dict["task_dependencies"]]
    data_dependencies = make_data_dependencies(task_dict["data_dependencies"])

    if 'task_mapping' in task_dict:
        task_mapping = device_from_string(task_dict["task_mapping"])
    else:
        task_mapping = None

    if 'task_order' in task_dict:
        task_order = int(task_dict["task_order"])
    else:
        task_order = 0
    
    return TaskInfo(task_id=task_id, task_runtime=task_runtime, task_dependencies=task_dependencies, data_dependencies=data_dependencies, task_mapping=task_mapping, task_order=task_order)
        
    
    



def test_generate_single_device_independent():

    cpu = Device(Architecture.CPU, 0)
    gpu = Device(Architecture.GPU, -1)

    gpu1 = Device(Architecture.GPU, 1)
    gpu2 = Device(Architecture.GPU, 2)


    #A few type checks:
    print("Is Tuple: ", isinstance(cpu, Tuple))
    print("Is Device: ", isinstance(cpu, Device))

    task_configs = TaskConfigs()
    task_configs.add((gpu1, gpu2), TaskConfig(task_time=10, gil_accesses=1))
    task_configs.add(cpu, TaskConfig(task_time=1000, gil_accesses=1))
    task_configs.add(gpu, TaskConfig(task_time=1000, gil_accesses=1))
    data_config = DataGraphConfig(pattern=1)
    config = SerialConfig(steps=2, task_config=task_configs, data_config=data_config, fixed_placement=False, n_devices=2)

    tasks, data = generate_single_device_serial(config)


    write_to_yaml(tasks, data)
    tasks, data = read_from_yaml()

    #write_to_pgraph(tasks, data)
    #tasks, data = read_from_pgraph()

    print(tasks)



    

    #print(tasks)


test_generate_single_device_independent()
