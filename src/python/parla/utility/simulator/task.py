import numpy as np
from fractions import Fraction

from data import PArray

class TaskHandle:
    """
    Used to create and initialize tasks.
    """
    # PArray name to object
    parray_name_to_obj = {}
    # Device id to object
    device_id_to_obj = {}
    datamove_task_meta_info = {}

    def __init__(self, task_id, runtime, dependency,
                 dependants, in_parray_names, out_parray_names,
                 datamove_tid_list):
        self.task_id = task_id
        self.runtime_info = runtime
        self.dependency = dependency
        self.dependants = dependants
        self.in_parray_names = in_parray_names
        self.out_parray_names = out_parray_names
        self.datamove_tid_list = datamove_tid_list

    @ staticmethod
    def set_parray_name_to_obj(parray_name_to_obj):
        TaskHandle.parray_name_to_obj = parray_name_to_obj

    @ staticmethod
    def set_device_id_to_obj(device_id_to_obj):
        TaskHandle.device_id_to_obj = device_id_to_obj

    @ staticmethod
    def set_datamove_task_meta_info(datamove_task_meta_info):
        TaskHandle.datamove_task_meta_info = datamove_task_meta_info

    def get_valid_devices(self):
        """
        Get a set of valid devices that are compatible with task.
        """
        contraint_set = list(self.runtime_info.keys())
        device_list = list(TaskHandle.device_id_to_obj.keys())
        n_gpus = len([d for d in device_list if d >= 0])

        all_idx, all_devices = zip(*TaskHandle.device_id_to_obj.items())

        # Add all devices
        valid_devices = set()
        for idx in contraint_set:
            if idx < 0:
                valid_devices.add(TaskHandle.device_id_to_obj[idx])
            if idx > 0:
                valid_devices.add(TaskHandle.device_id_to_obj[(idx-1)])

        # Add all GPU devices
        # TODO(hc):???
        if 0 in contraint_set:
            for device in all_devices:
                if not device.is_cpu():
                    valid_devices.add(device)

        return list(valid_devices)

    def get_task_config_on_device(self, device):
        #print("CHECK", self.runtime_info)

        if device.id < 0:
            if device.id in self.runtime_info:
                return self.runtime_info[device.id]

        shifted_idx = device.id + 1
        if shifted_idx > 0:
            if shifted_idx in self.runtime_info:
                return self.runtime_info[shifted_idx]

        if 0 in self.runtime_info:
            return self.runtime_info[0]

        return None

    def create_compute_task(self, device):
        """
        Create compute tasks.
        """

        configuration = self.get_task_config_on_device(device)

        compute_time = configuration[0]
        acus = configuration[1]
        gil_count = configuration[2]
        gil_time = configuration[3]
        memory = configuration[4]

        task_time = compute_time + gil_time*gil_count

        resources = {device.name: {"memory": memory, "acus": acus}}
        duration = task_time

        dependencies = self.dependency
        dependants = self.dependants

        in_parrays = [TaskHandle.parray_name_to_obj["D"+str(d)] \
                     for d in self.in_parray_names]
        out_parrays = [TaskHandle.parray_name_to_obj["D"+str(d)] \
                      for d in self.out_parray_names]

        target_device_for_movement = \
            {"D"+str(d): device for d in self.in_parray_names}

        return SyntheticTask(self.task_id, resources, duration, dependencies,
                             dependants, in_parrays, out_parrays,
                             target_device_for_movement)

    def create_movement_tasks(self, device):
        """
        Create data move tasks.
        """
        task_list = []

        for datamove_tid in self.datamove_tid_list:
            datamove_task_meta_info = \
                TaskHandle.datamove_task_meta_info[datamove_tid]
            resources = {device.name: {"memory": 0, "acus": Fraction(0)}}
            duration = 0

            dependencies = datamove_task_meta_info[1]
            dependants = [self.task_id]

            # PArrays to be read
            read_parray_names = datamove_task_meta_info[2]

            read_parrays = [TaskHandle.parray_name_to_obj[\
                            "D"+str(d)] for d in read_parray_names]
            target_device_for_movement = {"D"+str(d): device for
                                          d in read_parray_names}

            movement_task = SyntheticTask(
                datamove_tid, resources, duration, dependencies,
                dependants, read_parrays, [], target_device_for_movement)
            movement_task.set_to_data_movement()

            task_list.append(movement_task)

        return task_list


class SyntheticTask:
    """
    Task in Simulator.
    """
    taskspace = dict()

    def __init__(self, _name, _resources, _duration, _dependencies, _dependents,
                 _read_parrays, _write_parrays, _parray_targets, _order=None):
        self.name = _name
        self.resources = _resources
        self.duration = _duration

        # What tasks does this task depend on?
        self.dependencies = _dependencies

        # What tasks depend on this task?
        self.dependents = _dependents

        # Is this task ready?
        # 0: not ready, 1: ready, 2: completed
        self.status = 0

        # Priority order (if any)
        self.order = _order

        self.completion_time = -1.0

        # Set of data to read and write
        self.write_parrays = _write_parrays
        self.read_parrays = _read_parrays

        # Where does the data need to be located at task runtime?
        # (Needed for multidevice tasks)
        self.parray_targets = _parray_targets
        self.parray_sources = None

        # A device set where this task would run on.
        self.locations = list(self.resources.keys())

        self.is_movement = False
        self.unlock_flag = True
        self.evict_flag = True
        self.free_flag = True
        self.copy_flag = False
        self.is_redundant = False

        SyntheticTask.taskspace[self.name] = self

    def set_to_data_movement(self):
        self.is_movement = True
        self.unlock_flag = False
        self.free_flag = False
        self.copy_flag = True

    def make_evict(self):
        self.evict_flag = False

    # TODO(hc): instead of updating status, dependency counters look better.
    """
    def add_dependent(self, dependent):
        self.dependents.append(dependent)
        for dependent in self.dependents:
            dependent.set_notready_status()

    def add_dependency(self, dependency):
        # TODO(hc): if dependency has been already completed, status should not
        # be set to 0.
        self.dependencies.append(dependency)
        self.status = 0
    """

    def convert_tasknames_to_objs(self, taskspace):
        """
        Initially, all tasks are read from a networkx graph file and 
        are stored in string as their names.
        This function converts those names to task objects.
        """
        self.dependencies = [taskspace[dependency]
                             for dependency in self.dependencies]
        self.dependents = [taskspace[dependent]
                           for dependent in self.dependents]

    # TODO(hc): should use counters, intead of setting status..
    def update_status(self):
        self.status = 0
        # print("Updating Status of",  self.name, self.dependencies)
        for dependency in self.dependencies:
            if dependency.status != 2:
                self.status = 0
                return
        self.status = 1

    def check_status(self):
        """
        Check ready-related status.
        """
        return self.status

    def start(self):

        # If task is moving parrays, then update corresponding parrays
        # and device data pools
        if self.is_movement:
            assert(self.parray_sources is not None)
            for read_parray in self.read_parrays:
                read_parray.start_prefetch(
                    self, self.dependents, [self.parray_targets[read_parray.name]])
                # print("Prefetch Check: ", read_parray)

        # Lock all used parray and their source devices (if any)
        self.lock_parray()

        # Perform read on used PArrays
        # (So, those are moved to proper target devices)
        # (Decrements prefetch counter if not movement)
        for parray in self.read_parrays:
            target_device = self.parray_targets[parray.name]
            parray.read(self, [target_device], self.is_movement)

        # Perform write on used data:
        # 1. (causes eviction in the data pool for all others)
        # 2. (decrements prefetch counter)
        # NOTE: Data movement tasks should NOT be writing data, list should be empty.
        for parray in self.write_parrays:
            target_device = self.parray_targets[parray.name]
            parray.write(self, [target_device], self.is_movement)

    def finish(self):

        # Update status to completed
        self.status = 2

        # Propogate status to dependents
        for dependent in self.dependents:
            dependent.update_status()

        # Unlock all used data and their sources
        self.unlock_data()

        # If task has moved data, then update corresponding data blocks and device data pools
        if self.is_movement:
            for data in self.read_parrays:
                data.finish_prefetch(self, self.dependents, [
                    self.parray_targets[data.name]])

            # NOTE: Data movement tasks should NOT have any write data
            # for data in self.write_parrays:
            #    data.prefetch(self.dependents, [self.parray_sources[data.name]])

    def find_source_device(self, data, pool, required=False):
        success_flag = True

        # Find source
        target_device = self.parray_targets[data.name]
        source_device = data.choose_source_device(
            target_device, pool, required=required)
        # print("The best free connection is: ", source_device.name)
        self.parray_sources[data.name] = source_device

        if source_device is None:
            success_flag = False

        return success_flag

    def check_valid_parray(self, pool, required=False):
        """
        Check if operand PArrays exist on any of the devices.
        """
        self.parray_sources = dict()
        success_flag = True

        for parray in self.read_parrays:
            if not self.find_source_device(parray, pool, required=required):
                success_flag = False

        for parray in self.write_parrays:
            if not self.find_source_device(parray, pool, required=required):
                success_flag = False

        return success_flag

    def compute_movement_size(self):
        """
        Compute the total memory size of data movement for input/output
        PArrays.
        """
        sizes = []

        for parray in self.read_parrays:
            device_target = self.parray_targets[parray]
            size = device_target.get_nonlocal_size(parray)
            sizes.append(size)

        for parray in self.write_parrays:
            device_target = self.parray_targets[parray]
            size = device_target.get_nonlocal_size(parray)
            sizes.append(size)

        return sum(sizes)

    def check_data_locations(self):
        """
        Check if all data is located at the correct location to run the task
        """
        success_flag = True

        for parray_name, device in self.parray_targets.items():
            parray = PArray.dataspace[parray_name]

            if not parray.valid(device):
                success_flag = False
                break

        return success_flag

    def check_resources(self, pool):
        """
        Check if all resources and communication links are available to
        run the task.
        Check if all data is located at the correct location to run the task
        (if not movement).
        """
        success_flag = True

        # Make sure all data transfers can fit on the device
        if self.is_movement:

            for parray in self.read_parrays:
                device_target = self.parray_targets[parray.name]
                success_flag = success_flag and device_target.check_fit(parray)

            # NOTE: This should be empty
            for parray in self.write_parrays:
                device_target = self.parray_targets[parray.name]
                success_flag = success_flag and device_target.check_fit(parray)

        if not success_flag:
            return False

        # Make sure data transfer can be scheduled right now
        if self.is_movement:
            # Try to assign all data movement to active links
            success_flag = success_flag and self.check_valid_parray(
                pool, required=True)
            # print("Data Sources", self.parray_sources)
        else:
            # Make sure all data is already loaded
            success_flag = success_flag and self.check_data_locations()

            if not success_flag:
                raise Exception(
                    f"Data missing on device before task start. Task: {self.name}")

        # Make sure enough resources on the device are available to run the task
        if success_flag:
            return pool.check_resources(self.resources)
        else:
            return False

    def reserve_resources(self, pool):
        """
        Reserve resources and communication links for the task.
        """
        pool.reserve_resources(self.resources)

        if self.parray_sources:
            self.reserved = pool.topology.reserve_communication(
                self.parray_targets, self.parray_sources)

            if (len(self.reserved) == 0) and self.is_movement:
                self.is_redundant = True

    def free_resources(self, pool):
        """
        Free resources and communication links for the task.
        """
        if not self.is_movement:
            pool.free_resources(self.resources)

        if self.parray_sources:
            pool.topology.free_communication(
                self.parray_targets, self.parray_sources, self.reserved)

    def estimate_time(self, current_time, pool):
        # TODO: For data movement and eviction, compute from data movement time

        transfer_time = 0
        if self.is_movement:
            transfer_time = pool.topology.compute_transfer_time(
                current_time, self.parray_targets, self.parray_sources)

        # print(self.name, "Transfer Time:", transfer_time - current_time)

        # TODO: This assumes compute and data movement tasks are disjoint
        if self.is_movement:
            self.completion_time = transfer_time
        else:
            self.completion_time = current_time + self.duration

        assert(self.completion_time >= 0)
        assert(self.completion_time < float(np.inf))

        return self.completion_time

    def lock_parray(self):
        # print("Locking Data for Task:", self.name)

        joint_parrays = (set(self.read_parrays) | set(self.write_parrays))
        for parray in joint_parrays:
            to_lock = []

            device_target = self.parray_targets[parray.name]
            to_lock.append(device_target)

            if self.parray_sources:
                device_source = self.parray_sources[parray.name]
                assert(device_source is not None)
                if not (device_target == device_source):
                    # Source and target devices both should be locked
                    # for data transfer.
                    to_lock.append(device_source)
            # All devices using this PArray increase used counters.
            parray.acquire(to_lock)


    def unlock_data(self):

        joint_parrays = (set(self.read_parrays) | set(self.write_parrays))

        for parray in joint_parrays:
            to_unlock = []

            device_target = self.parray_targets[parray.name]
            to_unlock.append(device_target)

            if self.parray_sources:
                device_source = self.parray_sources[parray.name]
                assert(device_source is not None)
                if not (device_source == device_target):
                    to_unlock.append(device_source)

            parray.release(to_unlock)

        """
        for data in self.write_parrays:

            to_unlock = []

            device_target = self.parray_targets[data.name]
            to_unlock.append(device_target)

            if self.parray_sources:
                device_source = self.parray_sources[data.name]
                assert(device_source is not None)
                if not (device_source == device_target):
                    to_unlock.append(device_source)

            data.unlock(to_unlock)
        """

    def __str__(self):
        return f"Task: {self.name} | \n\t Dependencies: {self.dependencies} | \n\t Resources: {self.resources} | \n\t Duration: {self.duration} | \n\t Data: {self.read_parrays} : {self.write_parrays} | \n\t Needed Configuration: {self.parray_targets}"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)

    def __lt__(self, other):
        # If task is running on device, use completion time order
        if self.completion_time > 0 and other.completion_time > 0:
            return self.completion_time < other.completion_time
        # Otherwise use priority order (user assigned)
        else:
            return self.order < other.order

    def __eq__(self, other):
        # Two tasks are equal if they have the same name
        return self.name == other.name
