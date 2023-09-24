import priority_queue

PriorityQueue = priority_queue.PriorityQueue


class EvictionPolicy:

    def apply(self, datapool, active_tasks, planned_tasks):
        # Assign or update priority to every item in datapool
        pass


class DevicePArrayPool:

    def __init__(self, _device, _policy):
        self.pool = list()
        self.device = _device
        self.policy = _policy

        self.total_memory = 0
        self.in_progress_memory = 0

    def add_data(self, data):
        # Check if data is already in the pool
        if data in self.pool:
            return

        # If space on device is available, add to list and consume resources
        # TODO(hc): where does check available device space?
        self.pool.append(data)

        self.update_memory()

        # Rank by priority
        # self.policy.apply(self.pool, None, None)
        # self.pool.sort(key=lambda x: getattr(
        #    x.locations[self.device], "priority"))

    def evict(self):
        return self.pool.pop()

    def delete_data(self, data):
        self.pool.remove(data)
        self.update_memory()

    def get_status(self, data_name):
        return search_attribute(self.pool, data_name, "name").get_status(self.device)

    def extract_list_by_status(self, status, compare=None):
        """
        Extract arrays in the specified status on the current device.
        `status` is one of "in_progress", "prefetched", "all", "used."
        """
        if compare is None:
            # If the returned status of data copy in the specified device
            # is > 0, then, the data is in the specified status.
            compare = self.default_compare

        if status == "all":
            return self.pool
        else:
            sublist = list()
            for data in self.pool:
                if compare(getattr(
                     data.status_per_device[self.device.name], status)):
                    sublist.append(data)
            return sublist

    def __get_sublist_size(self, sublist):
        """
        Get the total bytes of the arrays on `sublist.`
        """
        return sum([data.size for data in sublist])

    def get_total_bytes(self, status, compare=None):
        """
        Get the list of arrays in a specified status,
        and get the total bytes of that.
        """
        if compare is None:
            compare = self.default_compare

        return self.__get_sublist_size(
            self.extract_list_by_status(status, compare))

    def update_memory(self):
        """
        Update device's memory status.
        """
        self.total_memory = self.get_total_bytes("all")
        self.in_progress_memory = self.get_total_bytes("in_progress")

        self.device.update_parray_memory_usage()

    def evictable_memory(self, include_prefetched=True):
        # TODO(hc): not using yet
        """
        """
        size = 0
        for data in self.pool:
            if data.locations[self.device].prefetched and not include_prefetched:
                continue
            if data.locations[self.device.name].used:
                continue
            size += data.size
        return size

    def default_compare(self, val):
        return val > 0

    def __contains__(self, item):
        return item in self.pool

    def __getitem__(self, index):
        return self.pool[index]

    def __setitem__(self, index, value):
        # NOTE: DON'T EVER USE THIS. For debugging convience only.
        # Use add_data to append instead.
        self.pool[index] = value

    def __str__(self):
        return f"{[ d.name+repr(d.get_status(self.device)) for d in self.pool]}"

    def __repr__(self):
        return f"{[repr(x) for x in self.pool]}"


class SyntheticDevice:
    """
    Logical device
    """

    # Number of the devices (CPU or GPU)
    num_device = 0
    # Mapping between device name to device
    devicespace = dict()

    def __init__(self, _name, _id, _resources, num_copy_engines=2,
                 idx=None, policy=EvictionPolicy()):
        self.name = _name
        self.id = _id
        if idx is not None:
            self.idx = idx
        else:
            self.idx = SyntheticDevice.num_device
        SyntheticDevice.num_device += 1

        # Memory and acus
        self.resources = _resources
        # TODO(hc): Rename some variables as they are confusing
        # Running tasks
        self.active_tasks = PriorityQueue()
        # Mapped compute tasks
        self.planned_compute_tasks = PriorityQueue()
        # Mapped data movement tasks
        self.planned_movement_tasks = PriorityQueue()
        self.num_copy_engines = num_copy_engines

        SyntheticDevice.devicespace[self.name] = self
        # Active data
        self.active_parray_pool = DevicePArrayPool(self, policy)
        # Current PArray memory usage size
        self.parray_memory_usage = 0
        # Resource pool to query; initially it is None, but it is set
        # to a proper resource pool during HW topology initialization.
        self.resource_pool = None

    def get_current_resources(self):
        """
        Update available memory and return the pool.
        """
        # TODO(hc): why not update acu?
        current_resources = copy.deepcopy(self.resource_pool.pool[self.name])
        current_resources["memory"] = self.available_memory()
        return current_resources

    def used_memory(self):
        """
        Return used memory size.
        """
        if self.resource_pool is not None:
            if self.name in self.resource_pool.pool:
                # static device memory size - (
                return self.resources["memory"] -                     \
                       (self.resource_pool.pool[self.name]["memory"] -\
                        self.parray_memory_usage)
        else:
            return False


    def available_memory(self):
        """
        Return available memory size.
        """
        if self.resource_pool is not None:
            if self.name in self.resource_pool.pool:
                return self.resource_pool.pool[self.name]["memory"] -  \
                       self.parray_memory_usage
        else:
            return False

    def used_acus(self):
        """
        Return factor of the used acus.
        """
        if self.name in self.resource_pool.pool:
            return 1 - self.resource_pool.pool[self.name]["acus"]

    def available_acus(self, usage=False):
        """
        Return factor of the available acus.
        """
        if self.name in self.resource_pool.pool:
            return self.resource_pool.pool[self.name]["acus"]

    def update_parray_memory_usage(self):
        """
        Update the total parray memory usage.
        """
        self.parray_memory_usage = self.active_parray_pool.total_memory

        # There should always be enough memory to fit both
        # the running tasks AND the persistent data.
        if hasattr(self, "resource_pool") and self.resource_pool is not None:
            if self.name in self.resource_pool.pool:
                assert(self.parray_memory_usage <
                       self.resource_pool.pool[self.name]["memory"])

    def get_nonlocal_size(self, data):
        """
        Return the size of data that needs to be moved if not already on the
        device; slice is not supporting.
        """
        if data in self.active_parray_pool:
            return 0
        return data.size

    def check_fit(self, parray):
        """
        Check if the parray is already on the device
        """
        if parray in self.active_parray_pool:
            # TODO: Check if its not stale, i dunno.
            # I don't think theres a good reason to rerequest memory.
            # print("ITS ALREADY THERE (possibly in progress")
            return True
        if available_memory := self.available_memory():
            if parray.size <= available_memory:
                return True
        return False

    """
    def check_fit_with_eviction(self, data):
        if self.check_fit(data):
            return True

        if available_memory := self.available_memory():
            if evictable_memory := self.evictable_memory():
                if data.size <= available_memory + evictable_memory:
                    return True
        return False

    def evictable_memory(self, include_prefetched=True):
        return self.active_parray_pool.evictable_memory(include_prefetched)

    def evict_best(self):
        return self.active_parray_pool.evict()
    """

    def resident_memory(self, in_progress=True):
        """
        Get the total memory size of the resident parrays.
        """
        if in_progress:
            return self.active_parray_pool.total_memory - \
                   self.active_parray_pool.in_progress_memory
        else:
            return self.active_parray_pool.total_memory

    def set_concurrent_copies(self, num_copy_engines):
        self.num_copy_engines = num_copy_engines

    def count_active_tasks(self, type):
        if type == "all":
            return len(self.active_tasks.queue)
        elif type == "all_useful":
            # TODO(hc): idk what it is
            return len([task for task in self.active_tasks.queue if task[1].is_redundant == False])
        elif type == "compute":
            return len([task for task in self.active_tasks.queue if task[1].is_movement == False])
        elif type == "movement":
            return len([task for task in self.active_tasks.queue if task[1].is_movement == True])
        elif type == "copy":
            return len([task for task in self.active_tasks.queue if (task[1].is_movement and not task[1].is_redundant)])

    def push_planned_task(self, _task):
        """
        Push planned task to a queue.
        """
        if _task.is_movement:
            self.planned_movement_tasks.put(_task)
        else:
            self.planned_compute_tasks.put(_task)
        return True

    def push_local_active_task(self, task):
        self.active_tasks.put((task.completion_time, task))
        return True

    def pop_local_active_task(self):
        return self.active_tasks.get()

    def delete_data(self, data):
        return self.active_parray_pool.delete_data(data)

    def add_data(self, data):
        return self.active_parray_pool.add_data(data)

    def is_cpu(self):
        return self.id < 0

    def __eq__(self, other):
        return self.name == other.name

    def __lt__(self, other):
        return self.id < other.id

    def __repr__(self):
        return str(self.name)

    def __str__(self):
        if self.resource_pool:
            return f"Device {self.name} | \n\t Planned Data Tasks: {self.planned_movement_tasks.queue} | \n\t Planned Compute Tasks: {self.planned_compute_tasks.queue} | \n\t Active Tasks: {self.active_tasks.queue} | \n\t Resources: {self.resource_pool.pool[self.name]} | \n\t Memory: {self.available_memory()} | \n\t Active Data: {self.active_parray_pool}"
        else:
            return f"Device {self.name} | \n\t Planned Tasks: {self.planned_tasks.queue} | \n\t Active Tasks: {self.active_tasks.queue} | \n\t Resources: {self.resources}"

    def __hash__(self):
        return hash(self.name)
