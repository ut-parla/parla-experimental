from ..types import *
from ..load import *
from .task import *
from .data import *
from .device import *


def data_from_task(task: TaskInfo, access: AccessType) -> List[DataID]:
    return [d.id for d in task.data_dependencies[access]]


def find_writer_bfs(graph: TaskMap, node: TaskID, target: DataID) -> TaskID | DataID:
    """
    Return last task to touch the data.
    @param graph: TaskMap (TaskID -> TaskInfo)
    @param node: TaskID to start search from
    @param target: DataID to search for
    """

    queue = []
    visited = []
    visited.append(node)
    queue.append(node)

    while queue:
        s = queue.pop(0)
        # print("Visiting: ", graph[s])
        for neighbor_id in graph[s].dependencies:
            neighbor = graph[neighbor_id]
            writes_to = data_from_task(neighbor, AccessType.WRITE)

            if target in writes_to:
                return neighbor_id if neighbor_id != node else target

            if neighbor_id not in visited:
                visited.append(neighbor.id)
                queue.append(neighbor.id)

    return target


DataWriter = Dict[DataID, TaskID | DataID]
DataWriters = Dict[TaskID, DataWriter]


def most_recent_writer(graph: TaskMap, task: TaskInfo) -> DataWriter:
    """
    For each of a tasks inputs, return the most recent writer.
    If this is the first task to write the data, return the DataID itself.
    """

    read_data = data_from_task(task, AccessType.READ)
    write_data = data_from_task(task, AccessType.READ_WRITE)

    read_data = read_data + write_data
    touches = set(read_data)

    recent_writer = dict()

    for target in touches:
        print("Finding writer for: ", target)
        recent_writer[target] = find_writer_bfs(graph, task.id, target)

    return recent_writer


def find_recent_writers(graph: TaskMap) -> DataWriters:
    """
    For each task, find the most recent writer for each of its inputs.
    """
    recent_writers = dict()

    for task in graph.values():
        recent_writers[task.id] = most_recent_writer(graph, task)

    return recent_writers


SimulatedTaskMap = Dict[TaskID, SimulatedTask]
SimulatedComputeTaskMap = Dict[TaskID, SimulatedComputeTask]
SimulatedDataTaskMap = Dict[TaskID, SimulatedDataTask]


def create_compute_tasks(graph: TaskMap) -> SimulatedComputeTaskMap:
    """
    Create compute tasks for each task in the graph.
    """
    compute_tasks = dict()

    for task in graph.values():
        compute_tasks[task.id] = SimulatedComputeTask(task.id, task)

    return compute_tasks


def create_data_tasks(graph: SimulatedComputeTaskMap, recent_writers: DataWriters) -> SimulatedDataTaskMap:
    """
    Create data tasks for each data item in the task.
    """
    data_tasks = dict()

    for task in graph.values():
        task_info = task.info
        recent_writer = recent_writers[task_info.id]
        for i, (data, writer) in enumerate(recent_writer.items()):
            dependencies = [writer] if isinstance(writer, TaskID) else []
            dependents = [task_info.id]

            data_task_id = TaskID(taskspace=f"{task_info.id}.data", task_idx=i)
            runtime = TaskRuntimeInfo(task_time=0)
            data_info = TaskDataInfo(read=[DataAccess(id=data)])
            data_task_info = TaskInfo(id=data_task_id, dependencies=dependencies,
                                      runtime=runtime, data_dependencies=data_info)

        data_task = SimulatedDataTask(name=data_task_id, info=data_task_info)
        data_tasks[data_task_id] = data_task
        task.add_data_dependency(data_task_id)

    return data_tasks


def create_task_graph(graph: TaskMap) -> Tuple[SimulatedComputeTaskMap, SimulatedDataTaskMap]:
    """
    Create a task graph from a task map.
    """
    compute_tasks = create_compute_tasks(graph)

    from rich import print
    print(compute_tasks)

    recent_writers = find_recent_writers(graph)

    data_tasks = create_data_tasks(compute_tasks, recent_writers)

    #print(data_tasks)

    return compute_tasks, data_tasks


def read_graph(graph_name: str) -> Tuple[SimulatedComputeTaskMap, SimulatedDataTaskMap, DataMap]:
    tasks = read_tasks_from_yaml(graph_name)
    data = read_data_from_yaml(graph_name)

    compute_tasks, data_tasks = create_task_graph(tasks)

    return compute_tasks, data_tasks, data
