from .types import *
import re

#########################################
# Log Parsing States
#########################################


class LogState(IntEnum):
    """
    Specifies the meaning of a log line. Used for parsing the log file.
    """
    ADDING_DEPENDENCIES = 0
    ADD_CONSTRAINT = 1
    ASSIGNED_TASK = 2
    START_TASK = 3
    RUNAHEAD_TASK = 4
    NOTIFY_DEPENDENTS = 6
    COMPLETED_TASK = 6
    UNKNOWN = 7

task_filter = re.compile(r'InnerTask\{ .*? \}')


def get_time(line: str) -> int:
    logged_time = line.split('>>')[0].strip().strip("\`").strip('[]')
    return int(logged_time)


def check_log_line(line: str) -> int:
    if "Running task" in line:
        return LogState.START_TASK
    elif "Notified dependents" in line:
        return LogState.NOTIFY_DEPENDENTS
    elif "Assigned " in line:
        return LogState.ASSIGNED_TASK
    elif "Runahead task" in line:
        return LogState.RUNAHEAD_TASK
    elif "Completed task" in line:
        return LogState.COMPLETED_TASK
    elif "Adding dependencies" in line:
        return LogState.ADDING_DEPENDENCIES
    elif "has constraints" in line:
        return LogState.ADD_CONSTRAINT
    else:
        return LogState.UNKNOWN


def convert_task_id(task_id: str, instance: int = 0) -> TaskID:
    id = task_id.strip().split('_')
    taskspace = id[0]
    task_idx = tuple([int(i) for i in id[1:]])
    return TaskID(taskspace, task_idx, int(instance))


def get_task_properties(line: str):
    message = line.split('>>')[1].strip()
    tasks = re.findall(task_filter, message)
    tprops = []
    for task in tasks:
        properties = {}
        task = task.strip('InnerTask{').strip('}').strip()
        task_properties = task.split(',')
        for prop in task_properties:
            prop_name, prop_value = prop.strip().split(':')
            properties[prop_name] = prop_value.strip()

        # If ".dm." is in the task name, ignore it since
        # this is a data movement task.
        # TODO(hc): we may need to verify data movemnt task too.
        if ".dm." in properties['name']:
            continue

        properties['name'] = convert_task_id(
            properties['name'], properties['instance'])

        tprops.append(properties)

    return tprops


def parse_blog(filename: str = 'parla.blog') -> Tuple[Dict[TaskID, TaskTime],  Dict[TaskID, List[TaskID]]]:

    try:
        result = subprocess.run(
            ['bread', '-s', r"-f `[%r] >> %m`", filename], stdout=subprocess.PIPE)

        output = result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        raise Exception(e.output)

    output = output.splitlines()

    task_start_times = {}
    task_runahead_times = {}
    task_notify_times = {}
    task_end_times = {}
    task_assigned_times = {}

    task_start_order = []
    task_end_order = []
    task_runahead_order = []

    task_times = {}
    task_states = defaultdict(list)

    task_dependencies = {}

    final_instance_map = {}

    for line in output:
        line_type = check_log_line(line)
        if line_type == LogState.START_TASK:
            start_time = get_time(line)
            task_properties = get_task_properties(line)

            if task_properties[0]["is_data_task"] == "1":
                continue

            task_properties = task_properties[0]

            task_start_times[task_properties['name']] = start_time
            task_start_order.append(task_properties['name'])

            current_name = task_properties['name']

            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            if base_name in final_instance_map:
                if current_name.instance > final_instance_map[base_name].instance:
                    final_instance_map[base_name] = current_name
            else:
                # if current_name.instance > 0:
                #    raise RuntimeError(
                #        "Instance number is not 0 for first instance of task")
                final_instance_map[base_name] = base_name

        elif line_type == LogState.RUNAHEAD_TASK:
            runahead_time = get_time(line)
            task_properties = get_task_properties(line)

            if task_properties[0]["is_data_task"] == "1":
                continue

            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_runahead_times[base_name] = runahead_time
            task_runahead_order.append(base_name)

        elif line_type == LogState.COMPLETED_TASK:
            end_time = get_time(line)
            task_properties = get_task_properties(line)

            if task_properties[0]["is_data_task"] == "1":
                continue

            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_end_times[base_name] = end_time
            task_end_order.append(base_name)

        elif line_type == LogState.NOTIFY_DEPENDENTS:
            notify_time = get_time(line)
            task_properties = get_task_properties(line)

            if task_properties[0]["is_data_task"] == "1":
                continue

            notifying_task = task_properties[0]
            current_name = notifying_task['name']
            current_state = notifying_task['get_state']
            instance = notifying_task['instance']

            if int(instance) > 0:
                base_name = TaskID(current_name.taskspace,
                                   current_name.task_idx,
                                   0)
                task_states[base_name] += [current_state]

            task_states[current_name] += [current_state]

        elif line_type == LogState.ASSIGNED_TASK:
            assigned_time = get_time(line)
            task_properties = get_task_properties(line)

            if task_properties[0]["is_data_task"] == "1":
                continue

            task_properties = task_properties[0]

            current_name = task_properties['name']
            base_name = TaskID(current_name.taskspace,
                               current_name.task_idx,
                               0)

            task_assigned_times[base_name] = assigned_time

        elif line_type == LogState.ADDING_DEPENDENCIES:
            task_properties = get_task_properties(line)

            if task_properties[0]["is_data_task"] == "1":
                continue

            current_task = task_properties[0]['name']
            current_dependencies = []

            for d in task_properties[1:]:
                dependency = d['name']
                current_dependencies.append(dependency)

            task_dependencies[current_task] = current_dependencies

    for task in task_end_times:
        assigned_t = task_assigned_times[task]
        start_t = task_start_times[task]
        # end_t = task_end_times[task]
        end_t = task_end_times[task]
        duration = end_t - start_t
        task_times[task] = TaskTime(assigned_t, start_t, end_t, duration)

    return task_times, task_dependencies, task_states
