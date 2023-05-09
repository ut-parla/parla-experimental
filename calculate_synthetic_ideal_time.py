import numpy as np

def independent_ideal_time(comp_time: int, num_gpus: int):
    """ The ideal time of the independent tasks 
        can be done when all the tasks are distributed
        to each GPU in a balanced manner. """
    return comp_time / float(num_gpus)


def reduction_ideal_time(comp_time: int, num_gpus: int, branch: int, levels: int):
    """ The ideal time of the reduction task graph
        is the maximum value among (the number of tasks * task's execution time)
        for each GPU. """
    # To store the total execution time for each GPU and to find the
    # maximum execution time among them.
    total_comp_time_np = np.zeros(num_gpus)
    for i in range(levels):
        total_in_level = branch ** i
        segment = total_in_level / num_gpus
        for j in range(total_in_level):
            device = int(j // segment)
            total_comp_time_np[device] += comp_time
    #print(total_comp_time_np)
    return total_comp_time_np.max(0)


def reduction_scatter_ideal_time(comp_time: int, num_gpus: int, num_tasks: int, levels:int):
    """ The ideal time of the reduction-scatter task graph
        is the maximum value among (the number of tasks * task's execution time)
        for each GPU. """
    total_comp_time_np = np.zeros(num_gpus)
    num_bridge_tasks = levels // 2
    num_bridge_tasks += 1 if (levels % 2 > 0) else  0
    num_bulk_tasks = (num_tasks - num_bridge_tasks)
    num_levels_for_bulk_tasks = levels // 2 + 1
    num_bulk_tasks_per_level = num_bulk_tasks // num_levels_for_bulk_tasks
    num_bulk_tasks_last_level = (num_bulk_tasks % num_levels_for_bulk_tasks) + num_bulk_tasks_per_level
    num_bulk_tasks_per_gpu = num_bulk_tasks_per_level // num_gpus
    task_id = 0
    bridge_task_dev_id = 0
    for l in range(levels + 1):
        if levels % 2 > 0:
            l_num_bulk_tasks = num_bulk_tasks_per_level if l < (levels - 1) else num_bulk_tasks_last_level
        else:
            l_num_bulk_tasks = num_bulk_tasks_per_level if l < levels else num_bulk_tasks_last_level
        if l % 2 > 0: # Bridge task condition
            total_comp_time_np[bridge_task_dev_id] += comp_time
            #print("bridge:", bridge_task_dev_id, ", comp time:", comp_time)
            bridge_task_dev_id += 1
            if bridge_task_dev_id == num_gpus:
                bridge_task_dev_id = 0
            task_id += 1
        else: # Bulk tasks condition
            bulk_task_id_per_gpu = 0
            bulk_task_dev_id = 0
            for bulk_task_id in range(l_num_bulk_tasks):
                total_comp_time_np[bulk_task_dev_id] += comp_time
                #print("bulk:", bulk_task_dev_id, ", comp time:", comp_time)
                l_num_bulk_tasks_per_gpu = l_num_bulk_tasks // num_gpus
                if l_num_bulk_tasks % num_gpus >= bulk_task_dev_id:
                    l_num_bulk_tasks_per_gpu += 1
                bulk_task_id_per_gpu += 1
                if bulk_task_id_per_gpu == l_num_bulk_tasks_per_gpu:
                    bulk_task_id_per_gpu = 0
                    bulk_task_dev_id += 1
                    if bulk_task_dev_id == num_gpus:
                        bulk_task_dev_id = 0
                task_id += 1
    #print(total_comp_time_np)
    return total_comp_time_np.max(0)


if __name__ == "__main__":
  print(independent_ideal_time(100, 4))
  print(reduction_ideal_time(100, 4, 2, 5))
  print(reduction_scatter_ideal_time(100, 4, 10, 4))
