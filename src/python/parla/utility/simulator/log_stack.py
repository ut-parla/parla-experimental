import copy

from collections import namedtuple

class LogDict:

    def __init__(self, graph, log_level=1):
        # Logs of tasks
        task_logs = dict()
        TaskLog = namedtuple("State", "status device start_time completion_time")
        # Initial task state logs
        for task in graph.nodes():
            task_logs[task] = TaskLog(0, None, None, None)
        # This dictionary manages not only task, but also
        # device and data logs. First, it stores task logs
        self.log_dict = task_logs
        # It logs (time, [task | parray | device] state logs)
        self.log_list = []
        self.log_list.append((0.0, task_logs))
        # Log level = {0: save minimal logs, 1: save maximal logs using
        # deep copy of objects; needed for plotting functions}
        # TODO(hc): change 0 or 1 to enum?
        self.log_level = log_level

    def log_device(self, device):
        """
        Log device related states.
        """
        if self.log_level == 1:
            # Log device object and its state.
            self.log_dict[device.name] = copy.deepcopy(device)
        else:
            # Log only resource state.
            self.log_dict[device.name] = device.get_current_resources()

    def log_parray(self, parray):
        """
        Log PArray related states.
        """
        if self.log_level == 1:
            # Log parray object and its state.
            self.log_dict[parray.name] = copy.deepcopy(parray)
        else:
            pass
            #self.log_dict[data.name] = copy.deepcopy(data.locations)

    def set_task_log(self, task, type, value):
        """
        Update task state log.
        """
        self.log_dict[task] = self.log_dict[task]._replace(
            **{type: value})

    def get_task_log(self, task, type):
        """
        Return task state log.
        """
        return getattr(self.log_dict[task], type)

    def advance_time(self, current_time):
        """
        Advance timer of logs.
        """
        self.log_list.append((current_time, self.log_dict))
        self.log_dict = copy.deepcopy(self.log_dict)

    def get_logs_with_time(self, time):
        """
        Return logs with time.
        """
        import bisect
        times, states = zip(*self.log_list)
        idx = bisect.bisect_left(times, time)
        return states[idx]

    def __getitem__(self, idx):
        return self.log_list[idx]

    def unpack(self):
        return zip(*self.log_list)

