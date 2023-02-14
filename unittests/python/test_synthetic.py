import pytest

from parla.utility.graphs import DeviceType, TaskConfig, TaskConfigs, SerialConfig, RunConfig
from parla.utility.graphs import read_pgraph, parse_blog
from parla.utility.graphs import generate_serial_graph, shuffle_tasks

from parla.utility.execute import run as run_graph
from parla.utility.execute import verify_order, verify_dependencies, verify_complete, verify_time
from parla.utility.execute import timeout
# from parla.utility.execute import timeout

import tempfile
import os
from ast import literal_eval as make_tuple


def test_read():
    """
    Test to test graph file format. (Used to save task graphs with constraints to text files)
    """

    s = make_tuple("('D', (1))")
    print(s)

    with tempfile.TemporaryDirectory() as tempdir:
        tmpfilepath = os.path.join(tempdir, 'test.gph')
        with open(tmpfilepath, 'w') as tmpfile:
            print(
                "{1 : -1}, {1 : -1}, {1 : -1}, {1 : -1}, {1:  -1}, {1: -1}, {1:  -1}, {1: -1}", file=tmpfile)
            print(
                "0 | {0 : 1, 0.5, 0, 200, 10}|  | 1  :  : 0", file=tmpfile)
            print(
                "('D', 1) | {0 : 1, 0.5, 0, 200, 10}, {1 : 1, 0.5, 0, 200, 10} | ('T', 0) : 1 | 1  :  : 0", file=tmpfile)

            print(
                "('D', (2, 3)) | {0 : 1, 0.5, 0, 200, 10}, {1 : 1, 0.5, 0, 200, 10} |   | 1  :  : 0", file=tmpfile)

        # write a test graph
        data_config, tasks = read_pgraph(tmpfilepath)

        # G[0] is a list of data objects
        assert (len(data_config) == 8)

        task_iter = iter(tasks)

        # the first task
        task = next(task_iter)
        task_info = tasks[task]
        # print(task_info)
        assert (task_info.task_id == ("T", (0,), 0))
        assert (task_info.task_runtime == {(0,): (1, 0.5, 0, 200, 10)})
        assert (task_info.task_dependencies == [])
        assert (task_info.data_dependencies == ([1], [], [0]))

        # the second task
        task = next(task_iter)
        task_info = tasks[task]
        assert (task_info.task_id == ("D", (1,), 0))
        assert (task_info.task_runtime == {
                (0,): (1, 0.5, 0, 200, 10), (1,): (1, 0.5, 0, 200, 10)})
        assert (task_info.task_dependencies == [
                ("T", (0,), 0), ("D", (1,), 0)])
        assert (task_info.data_dependencies == ([1], [], [0]))

        # the second task
        task = next(task_iter)
        task_info = tasks[task]
        assert (task_info.task_id == ("D", (2, 3), 0))
        assert (task_info.task_runtime == {
                (0,): (1, 0.5, 0, 200, 10), (1,): (1, 0.5, 0, 200, 10)})
        assert (task_info.task_dependencies == [])
        assert (task_info.data_dependencies == ([1], [], [0]))


def test_serial():
    """
    Test a chain of serial tasks.
    """
    with tempfile.TemporaryDirectory() as tempdir:

        tmpfilepath = os.path.join(tempdir, 'test_serial.graph')
        tmplogpath = os.path.join(tempdir, 'test_serial.blog')
        logfile = "test_serial.blog"

        with open(tmpfilepath, 'w') as tmpfile:
            max_time = 3
            n = 10
            task_time = 100000

            total_time_factor = 1.05
            local_task_factor = 1.5

            # write a test graph
            task_configs = TaskConfigs()
            task_configs.add(DeviceType.CPU_DEVICE, TaskConfig(
                task_time=task_time, gil_accesses=1))

            config = SerialConfig(
                steps=n, task_config=task_configs, dependency_count=3, chains=2)
            graph = generate_serial_graph(config)
            print(graph)
            tmpfile.write(graph)

        data_config, truth_graph = read_pgraph(tmpfilepath)

        run_config = RunConfig(
            outer_iterations=1, inner_iterations=1, verbose=False, logfile=logfile)

        @timeout(max_time)
        def run():
            timing = run_graph(truth_graph, data_config=data_config,
                               run_config=run_config)
            return timing

        timing = run()

        log_times, log_graph = parse_blog(logfile)

        # Verify that all tasks have run
        assert (verify_complete(log_graph, truth_graph))

        # Verify that all depdenencies have run (this would fail even if complete is true if the graph is wrong)
        assert (verify_dependencies(log_graph, truth_graph))

        # Verify that the tasks ran in the correct order
        assert (verify_order(log_times, truth_graph))

        # Verify that each task took about the right amount of time
        assert (verify_time(log_times, truth_graph, factor=local_task_factor))

        # Verify that the total time isn't too long
        assert (timing.mean < total_time_factor * n * task_time)
