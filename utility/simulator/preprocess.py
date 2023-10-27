from ..types import *
from ..load import *
from .task import *
from .data import *
from .device import *
import networkx as nx


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


def create_compute_tasks(graph: TaskMap) -> SimulatedComputeTaskMap:
    """
    Create compute tasks for each task in the graph.
    """
    compute_tasks = dict()

    for task in graph.values():
        compute_tasks[task.id] = SimulatedComputeTask(task.id, task)

    return compute_tasks


def create_data_tasks(
    graph: SimulatedComputeTaskMap, recent_writers: DataWriters
) -> SimulatedDataTaskMap:
    """
    Create data tasks for each data item in the task.
    """
    data_tasks = dict()

    for task in graph.values():
        task_info = task.info
        recent_writer = recent_writers[task_info.id]
        for i, (data, writer) in enumerate(recent_writer.items()):
            dependencies = [writer] if isinstance(writer, TaskID) else []

            data_task_id = TaskID(taskspace=f"{task_info.id}.data", task_idx=data.idx)

            runtime = TaskPlacementInfo()
            runtime.add(Device(Architecture.ANY, -1), TaskRuntimeInfo())
            data_info = TaskDataInfo(read=[DataAccess(id=data)])

            data_task_info = TaskInfo(
                id=data_task_id,
                dependencies=dependencies,
                runtime=runtime,
                data_dependencies=data_info,
            )

            data_task = SimulatedDataTask(name=data_task_id, info=data_task_info)
            data_tasks[data_task_id] = data_task
            task.add_data_dependency(data_task_id)

    return data_tasks


def filter_data_dependenices(task: SimulatedTask):
    data_info = task.info.data_dependencies
    read = data_info.read
    write = data_info.write
    read_write = data_info.read_write

    read_set = set([d for d in read]).union([d for d in read_write])
    write_set = set([d for d in write]).union([d for d in read_write])

    read = list(read_set)
    write = list(write_set)

    data_info.read = read
    data_info.write = write


def create_task_graph(
    graph: TaskMap, data=False
) -> Tuple[SimulatedComputeTaskMap, SimulatedDataTaskMap]:
    """
    Create a task graph from a task map.
    """
    compute_tasks = create_compute_tasks(graph)
    if data:
        recent_writers = find_recent_writers(graph)
        data_tasks = create_data_tasks(compute_tasks, recent_writers)
    else:
        data_tasks = False

    return compute_tasks, data_tasks


def read_graph(
    graph_name: str, data=False
) -> Tuple[List[TaskID], SimulatedTaskMap, DataMap]:
    tasks = read_tasks_from_yaml(graph_name)
    datamap = read_data_from_yaml(graph_name)

    compute_tasks, data_tasks = create_task_graph(tasks, data=False)
    if data:
        taskmap = {**compute_tasks, **data_tasks}
    else:
        taskmap = compute_tasks

    tasklist = list(compute_tasks.keys())

    populate_dependents(taskmap)

    return tasklist, taskmap, datamap


def build_networkx_graph(
    tasks: SimulatedTaskMap,
) -> Tuple[nx.DiGraph, Dict[TaskID, str]]:
    G = nx.DiGraph()
    labels = {}

    color_map = ["red", "blue"]

    for name, info in tasks.items():
        color_idx = 0 if isinstance(info, SimulatedComputeTask) else 1
        color = color_map[color_idx]

        G.add_node(name, label=name, color=color, info=info)
        for dependency in info.dependencies:
            G.add_edge(dependency, name, color=color)

        labels[name] = str(name)

    return G, labels


def apply_networkx_order(G: nx.DiGraph, tasks: SimulatedTaskMap) -> List[TaskID]:
    """
    Sort a graph by a topology, and return a valid order of the graph.
    """
    import networkx as nx

    nodes = list(nx.topological_sort(G))

    for i, node in enumerate(nodes):
        tasks[node].info.order = i

    return nodes


def sort_tasks_by_order(tasklist: List[TaskID], taskmap: SimulatedTaskMap):
    """
    Sort a list of tasks by their order in the taskmap.
    """
    return sorted(tasklist, key=lambda task: taskmap[task].info.order)


def populate_dependents(taskmap: SimulatedTaskMap):
    """
    Populate the dependents field of each task.
    @param taskmap: SimulatedTaskMap
    """
    for task in taskmap.values():
        for dependency in task.dependencies:
            taskmap[dependency].dependents.append(task.name)


def apply_mapping(taskmap: SimulatedTaskMap, device: Device):
    """
    Apply a mapping to a taskmap.
    @param taskmap: SimulatedTaskMap
    @param device: Device
    """
    for task in taskmap.values():
        task.info.mapping = device
