from .types import *
import yaml


def convert_to_dictionary(task_list: List[TaskInfo]) -> Dict[TaskID, TaskInfo]:
    """
    Converts a task list to a task graph dictionary
    """
    task_dict = dict()
    for task in task_list:
        task_dict[task.id] = task

    return task_dict


########################################
# YAML Write
########################################


def write_object_to_dict(obj):
    """
    Write a single task to an open YAML file
    """
    sub_dict = {}

    def is_base(x):
        return isinstance(x, (int, float, str, bool, type(None)))

    def is_base_str(x):
        return isinstance(x, (tuple, Architecture, Device, DataID))

    def is_base_value(x):
        return isinstance(x, (Decimal, Fraction))

    def unpack_values(values):
        if is_base_str(value):
            return str(value)
        elif is_base(value):
            return value
        elif is_base_value(value):
            return numeric_to_str(value)
        elif isinstance(value, TaskPlacementInfo):
            return write_object_to_dict(value.info)
        elif isinstance(value, list):
            return [write_object_to_dict(x) for x in value]
        else:
            return write_object_to_dict(value)

    if isinstance(obj, Dict):
        for key, value in obj.items():
            key = str(key)
            sub_dict[key] = unpack_values(value)
    elif is_base(obj):
        return obj
    else:
        for slot in obj.__slots__:
            value = getattr(obj, slot)
            sub_dict[slot] = unpack_values(value)

    return sub_dict


def write_data_to_yaml(data: Dict[DataID, DataInfo], basename: str = "graph"):
    """
    Write the data specifiers to a yaml file
    """

    datafile = basename + ".data.yaml"

    with open(datafile, "w") as file:
        data = [write_object_to_dict(data) for data in data.values()]
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)


def write_tasks_to_yaml(tasks: Dict[TaskID, TaskInfo], basename: str = "graph"):
    """
    Write the task graph to a yaml file
    """

    taskfile = basename + ".tasks.yaml"

    with open(taskfile, "w") as file:
        tasks = [write_object_to_dict(task) for task in tasks.values()]
        yaml.dump(tasks, file, default_flow_style=False, sort_keys=False)


def write_task_mapping_to_yaml(
    task_mapping: Dict[TaskID, Device | Tuple[Device]], basename: str = "graph"
):
    """
    Write the task -> device mapping to a yaml file
    """

    taskfile = basename + ".mapping.yaml"

    with open(taskfile, "w") as file:
        maplist = []
        for task_id, device in task_mapping.items():
            mapping_dict = {"id": task_id, "mapping": device}
            maplist.append(write_object_to_dict(mapping_dict))

        yaml.dump(maplist, file, default_flow_style=False, sort_keys=False)


def write_task_order_to_yaml(task_order: Dict[TaskID, int], basename: str = "graph"):
    """
    Write the task -> order mapping to a yaml file
    """

    taskfile = basename + ".order.yaml"

    with open(taskfile, "w") as file:
        maplist = []
        for task_id, order in task_order.items():
            mapping_dict = {"id": task_id, "order": order}
            maplist.append(write_object_to_dict(mapping_dict))

        yaml.dump(maplist, file, default_flow_style=False, sort_keys=False)


def write_to_yaml(
    tasks: Optional[Dict[TaskID, TaskInfo]] = None,
    data: Optional[Dict[int, DataInfo]] = None,
    mapping: Optional[Dict[TaskID, Device | Tuple[Device]]] = None,
    order: Optional[Dict[TaskID, int]] = None,
    basename: str = "graph",
):
    """
    Write the task graph to a yaml file
    """

    if tasks is not None:
        write_tasks_to_yaml(tasks, basename)

    if data is not None:
        write_data_to_yaml(data, basename)

    if mapping is not None:
        write_task_mapping_to_yaml(mapping, basename)

    if order is not None:
        write_task_order_to_yaml(order, basename)


########################################
# YAML Read
########################################


def read_tasks_from_dict(task_dict: Dict) -> TaskInfo:
    """
    Read a task from a dictionary
    """

    task_id = make_task_id_from_dict(task_dict["id"])

    task_runtime = make_task_placement_from_dict(task_dict["runtime"])
    task_dependencies = [
        make_task_id_from_dict(task) for task in task_dict["dependencies"]
    ]
    data_dependencies = make_data_dependencies_from_dict(task_dict["data_dependencies"])

    if "mapping" in task_dict:
        task_mapping = device_from_string(task_dict["mapping"])
    else:
        task_mapping = None

    if "order" in task_dict:
        task_order = int(task_dict["order"])
    else:
        task_order = 0

    return TaskInfo(
        id=task_id,
        runtime=task_runtime,
        dependencies=task_dependencies,
        data_dependencies=data_dependencies,
        mapping=task_mapping,
        order=task_order,
    )


def read_mapping_from_dict(mapping_dict: Dict) -> Tuple[TaskID, Device | Tuple[Device]]:
    """
    Read a task -> device mapping from a dictionary
    """

    task_id = make_task_id_from_dict(mapping_dict["id"])
    task_mapping = device_from_string(mapping_dict["mapping"])

    return task_id, task_mapping


def read_order_from_dict(order_dict: Dict) -> Tuple[TaskID, int]:
    """
    Read a task -> order mapping from a dictionary
    """

    task_id = make_task_id_from_dict(order_dict["id"])
    task_order = int(order_dict["order"])

    return task_id, task_order


def read_data_from_yaml(basename: str = "graph") -> Dict[DataID, DataInfo]:
    """
    Read the data specification from a yaml file
    """

    datafile = basename + ".data.yaml"

    try:
        data_dict = dict()
        with open(datafile, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)
            data = [make_data_info(data) for data in data]

        for data in data:
            data_dict[data.id] = data
    except FileNotFoundError:
        raise Warning(f"Could not find data file {datafile}")
        return None

    return data_dict


def read_tasks_from_yaml(basename: str = "graph") -> Dict[TaskID, TaskInfo]:
    """
    Read the task graph from a yaml file
    """
    taskfile = basename + ".tasks.yaml"

    try:
        task_dict = dict()
        with open(taskfile, "r") as file:
            tasks = yaml.load(file, Loader=yaml.FullLoader)
            tasks = [read_tasks_from_dict(task) for task in tasks]

        for task in tasks:
            task_dict[task.id] = task
    except FileNotFoundError:
        raise Warning(f"Could not find task file {taskfile}")
        return None

    return task_dict


def read_task_mapping_from_yaml(
    basename: str = "graph",
) -> Dict[TaskID, Device | Tuple[Device]]:
    """
    Read the task -> device mapping from a yaml file
    """
    taskfile = basename + ".mapping.yaml"

    try:
        task_mapping = dict()
        with open(taskfile, "r") as file:
            mappings = yaml.load(file, Loader=yaml.FullLoader)
            mappings = [read_mapping_from_dict(mapping) for mapping in mappings]

        for mapping in mappings:
            task_mapping[mapping.id] = mapping.mapping
    except FileNotFoundError:
        raise Warning(f"Could not find mapping file {taskfile}")
        return None

    return task_mapping


def read_task_order_from_yaml(basename: str = "graph") -> Dict[TaskID, int]:
    """
    Read the task -> order mapping from a yaml file
    """
    taskfile = basename + ".order.yaml"

    task_order = dict()
    try:
        with open(taskfile, "r") as file:
            orders = yaml.load(file, Loader=yaml.FullLoader)
            orders = [read_order_from_dict(order) for order in orders]

        for order in orders:
            task_order[order[0]] = order[1]

    except FileNotFoundError:
        raise Warning(f"Could not find order file {taskfile}")
        return None

    return task_order


def read_from_yaml(
    taskfile: Optional[str] = None, datafile: Optional[str] = None
) -> Tuple[Optional[Dict[TaskID, TaskInfo]], Optional[Dict[DataID, DataInfo]]]:
    """
    Read the task graph from a yaml file
    """

    if taskfile is None:
        tasks = None
    else:
        tasks = read_tasks_from_yaml(taskfile)

    if datafile is None:
        data = None
    else:
        data = read_data_from_yaml(datafile)

    return tasks, data


###########################################
# Legacy "PGRAPH" Parla Graph Format Write
###########################################


def write_to_pgraph(
    tasks: Dict[TaskID, TaskInfo], data: Dict[int, DataInfo], basename: str = "graph"
):
    """
    Write the task graph to a pgraph file
    """

    taskfile = basename + ".pgraph"

    def info_to_comma(info):
        comma = ", ".join([f"{getattr(info, slot)}" for slot in info.__slots__])
        return comma

    def unpack_runtime(
        runtime: TaskPlacementInfo
        | TaskRuntimeInfo
        | Dict[Device | Tuple[Device, ...], TaskRuntimeInfo]
    ):
        print("Unpacking runtime: ", type(runtime))
        if isinstance(runtime, TaskPlacementInfo):
            return unpack_runtime(runtime.info)
        elif isinstance(runtime, Dict):
            return ", ".join(
                [f"{{{device} : {unpack_runtime(r)}}}" for device, r in runtime.items()]
            )
        elif isinstance(runtime, TaskRuntimeInfo):
            return info_to_comma(runtime)
        elif isinstance(runtime, list):
            raise NotImplementedError(
                f"PGraph does not support lists of runtime configurations (Device configurations cannot vary by their index in the placement options)"
            )
        else:
            raise ValueError(f"Unknown runtime type {runtime}")

    def unpack_id(task_id):
        return f"('{task_id.taskspace}', {tuple(task_id.task_idx)})"

    def unpack_dependencies(dependencies: list[TaskID]):
        return ": ".join([f" {unpack_id(task_id)} " for task_id in dependencies])

    def unpack_data_dependencies(dependencies: TaskDataInfo):
        read_data = dependencies.read
        write_data = dependencies.write
        read_write_data = dependencies.read_write

        read_string = ", ".join([f"{data}" for data in read_data])
        write_string = ", ".join([f"{data}" for data in write_data])
        read_write_string = ", ".join([f"{data}" for data in read_write_data])

        return f"{read_string} : {write_string} : {read_write_string}"

    data_line = ", ".join(
        [f"{{{data.size} : {data.location}}}" for data in data.values()]
    )
    task_lines = []

    for task in tasks.values():
        task_id = unpack_id(task.id)
        task_runtime = unpack_runtime(task.runtime)
        task_dependencies = unpack_dependencies(task.dependencies)
        task_data_dependencies = unpack_data_dependencies(task.data_dependencies)

        task_line = f"{task_id} | {task_runtime} | {task_dependencies} | {task_data_dependencies}"
        task_lines.append(task_line)

    with open(taskfile, "w") as file:
        print(data_line, file=file)
        for task_line in task_lines:
            print(task_line, file=file)


##########################################
# Legacy "PGRAPH" Parla Graph Format Read
##########################################


def read_from_pgraph(
    basename: str = "graph",
) -> Tuple[Dict[TaskID, TaskInfo], Dict[int, DataInfo]]:
    task_dict = dict()
    data_dict = dict()

    filename = basename + ".pgraph"

    def extract_task_id(line: str) -> TaskID:
        try:
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
        except Exception as e:
            raise ValueError(f"Could not parse task id {line}: {e}")

    def extract_task_runtime(
        line: str,
    ) -> TaskPlacementInfo:
        try:
            line = line.strip()

            configurations = line.split("},")
            configurations = [
                config.strip().strip("{}").strip() for config in configurations
            ]
            task_runtime = {}
            for config in configurations:
                targets, details = config.split(":")
                targets = device_from_string(targets)

                details = [
                    numeric_from_str(detail.strip()) for detail in details.split(",")
                ]

                task_runtime[targets] = TaskRuntimeInfo(*details)

            task_runtime = TaskPlacementInfo(info=task_runtime)
            task_runtime.update()

            return task_runtime
        except Exception as e:
            raise ValueError(f"Could not parse task runtime {line}: {e}")

    def extract_task_dependencies(line: str) -> list[TaskID]:
        try:
            line = line.strip()
            dependencies = line.split(":")
            dependencies = [dependency.strip() for dependency in dependencies]

            if dependencies[0] == "":
                return []

            return [extract_task_id(dependency) for dependency in dependencies]
        except Exception as e:
            raise ValueError(f"Could not parse task dependencies {line}: {e}")

    def extract_data_dependencies(line: str) -> TaskDataInfo:
        try:
            line = line.strip()
            dependencies = line.split(":")
            dependencies = [dependency.strip() for dependency in dependencies]

            check_has = [
                (not dependency.isspace()) and (not dependency == "")
                for dependency in dependencies
            ]

            if not any(check_has):
                return TaskDataInfo([], [], [])

            if len(dependencies) > 3:
                raise ValueError(f"Too many data movement types {dependencies}")

            if len(dependencies) < 1 or dependencies[0].isspace() or not check_has[0]:
                read_data = []
            else:
                read_data = [
                    DataAccess(int(x.strip())) for x in dependencies[0].split(",")
                ]

            if len(dependencies) < 2 or dependencies[1].isspace() or not check_has[1]:
                write_data = []
            else:
                write_data = [
                    DataAccess(int(x.strip())) for x in dependencies[1].split(",")
                ]

            if len(dependencies) < 3 or dependencies[2].isspace() or not check_has[2]:
                read_write_data = []
            else:
                read_write_data = [
                    DataAccess(int(x.strip())) for x in dependencies[2].split(",")
                ]

            return TaskDataInfo(read_data, write_data, read_write_data)
        except Exception as e:
            raise ValueError(f"Could not parse data dependencies {line}: {e}")

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
            data_info = DataInfo(DataID((idx,)), data_size, data_location)
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

            task_info = TaskInfo(
                task_id, task_runtime, task_dependencies, task_data_dependencies
            )
            task_dict[task_id] = task_info

    return task_dict, data_dict
