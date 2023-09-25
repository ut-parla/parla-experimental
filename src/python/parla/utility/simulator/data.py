from device import SyntheticDevice

class PArrayStatus:
    def __init__(self, _stale=0, _used=0, _prefetch=0, _in_progress=False):
        self.stale = _stale
        self.used = _used
        self.prefetched = _prefetch
        self.in_progress = _in_progress

        # Tasks that depend on this data (that have been prefetched for)
        self.dependent_tasks = []

        # Tasks that are currently moving this data (e.g. prefetching)
        self.moving_tasks = []  # PriorityQueue()

        # Tasks that are removing this data (e.g. evicting from this device)
        self.eviction_tasks = []

    def __str__(self):
        return f"Data State :: stale: {self.stale} | used: {self.used} | prefetch: {self.prefetched} | in_progress: {self.in_progress}"

    def __repr__(self):
        return f"({self.stale}, {self.used}, {self.prefetched}, {self.in_progress})"

    def __hash__(self):
        return hash((self.stale, self.used, self.prefetched, self.in_progress))


class PArray:

    dataspace = dict()

    def __init__(self, _name, _size, device_list):
        self.name = _name
        self.size = _size

        # PArray status per device: e.g., task is depending on
        # parray, task is moving parray, etc.
        self.status_per_device = dict()

        for device in device_list:
            self.status_per_device[device.name] = PArrayStatus()

        # Must be set before use.
        PArray.dataspace[self.name] = self

        # Number of PArray transfer to a device.
        self.num_transfers = dict()

    def get_status(self, device):
        """
        Return the current PArray status on a specified device.
        """
        if device.name in self.status_per_device:
            return self.status_per_device[device.name]
        else:
            return False

    def add_transfer(self, device):
        """
        Increase the number of transfers on a specified device.
        """
        if device.name in self.num_transfers:
            self.num_transfers[device.name] += 1
        else:
            self.num_transfers[device.name] = 1

    def get_dependent_tasks(self, checking_device_list=None):
        """
        Return tasks that use this PArray.
        """
        dependent_tasks = []
        if checking_device_list is None:
            for device in self.status_per_device:
                dependent_tasks.extend(
                    self.status_per_device[device.name].dependent_tasks)
        else:
            for device in checking_device_list:
                dependent_tasks.extend(
                    self.status_per_device[device.name].dependent_tasks)
        return dependent_tasks

    def update_state(self, device, stale=0, used=0, prefetch=0):
        """
        Update the current PArray state.
        """
        if device.name in self.status_per_device:
            self.status_per_device[device.name].stale = stale
            self.status_per_device[device.name].used = used
            self.status_per_device[device.name].prefetched = prefetch
            return True
        return False

    def increment_state(self, device, stale=False, used=False, prefetch=False):
        """
        Increase counters for the current PArraystate.
        """
        if device.name in self.status_per_device:
            if stale:
                self.status_per_device[device.name].stale += 1
            if used:
                self.status_per_device[device.name].used += 1
            if prefetch:
                self.status_per_device[device.name].prefetched += 1
            return True
        return False

    def decrement_state(self, device, stale=False, used=False, prefetch=False):
        """
        Decrease counters for the current PArraystate.
        """
        if device.name in self.status_per_device:
            if stale:
                self.status_per_device[device.name].stale -= 1
            if used:
                self.status_per_device[device.name].used -= 1
            if prefetch:
                self.status_per_device[device.name].prefetched -= 1
            return True
        return False

    def update_status(self, device, state, value):
        """
        Update this PArray's status on a specified device.
        """
        if device.name in self.status_per_device:
            setattr(self.status_per_device[device.name], state, value)
            return True
        return False

    def query_status(self, device, property):
        """
        Query this PArray's status on a specified device.
        """
        if device.name in self.status_per_device:
            if hasattr(self.status_per_device[device.name], property):
                return getattr(self.status_per_device[device.name], property)
        return None

    def valid(self, device, allow_in_progress=False):
        """
        Return True if this device has a valid data copy.
        If `allow_in_progress` is True, moving data is considered
        as valid data.
        """
        if device.name in self.status_per_device:
            if self.status_per_device[device.name].stale:
                return False
            if not allow_in_progress and self.status_per_device[device.name].in_progress:
                return False
        else:
            return False
        return True

    def evict(self, device_name):
        """
        Release all staled PArrays from a specified device.
        """
        if device_name in self.status_per_device:
            if self.status_per_device[device_name].stale and \
               not self.status_per_device[device_name].used:

                # Delete from data table
                del self.status_per_device[device_name]

                # Delete from device list
                device = SyntheticDevice.devicespace[device_name]
                device.delete_data(self)

            elif self.status_per_device[device_name].stale and \
                 self.status_per_device[device_name].used:
                raise Exception("Invalid State. Data is stale and used.")

    def evict_stale(self):
        """
        Release all staled PArrays from all devices.
        """
        device_names = list(self.status_per_device.keys())
        state = list(self.status_per_device.values())
        for device_name, state in zip(device_names, state):
            # print("Eviction: ", device_name, state)
            if state.stale:
                self.evict(device_name)

    def create_copy_on_device(self, device, in_progress=True):
        """
        Create new PArray copy on a specified device.
        """
        if device.name not in self.status_per_device:
            self.status_per_device[device.name] = PArrayStatus(0, 0, 0, in_progress)

        # if in_progress and not self.status_per_device[device.name].in_progress:
        #    self.status_per_device[device.name].in_progress = True

    def release(self, device_list):
        """
        Release this PArray and decrement used counter.
        """
        for device in device_list:
            if device.name in self.status_per_device:
                print("Releasing data: ", self.name, device_list)
                self.status_per_device[device.name].used -= 1
                assert(self.status_per_device[device.name].used >= 0)
            else:
                raise Exception(
                    "Attempting to release data that is not on device.")

    def acquire(self, device_list):
        """
        Acquire this PArray and increment used counter.
        """
        # Increment used counter
        for device in device_list:
            if device.name in self.status_per_device:
                print("Acquiring data: ", self.name, device_list)
                self.status_per_device[device.name].used += 1
                assert(self.status_per_device[device.name].used >= 0)
            else:
                raise Exception(
                    "Attempting to acquire data that is not on device.")

    def start_prefetch(self, calling_task, task_list, device_list):
        """
        Start prefetching and update data and device states.
        """
        for device in device_list:

            # print("OLD: ", self)
            # Add to data table (Create state if not already there)
            self.create_copy_on_device(device, in_progress=True)
            # print("NEW: ", self)

            # Add to device pool
            # NOTE: Must be done after create_copy_on_device (because state is used to update the memory tracking)
            device.add_data(self)

            assert(device.name in self.status_per_device)

            # Increment prefetch counter
            # print("Incrementing prefetch counter", self, task_list)
            self.status_per_device[device.name].prefetched += 1
            self.status_per_device[device.name].dependent_tasks.extend(task_list)

            assert(calling_task.completion_time >= 0)
            # self.status_per_device[device.name].moving_tasks.put(
            #    (calling_task.completion_time, calling_task))
            self.status_per_device[device.name].moving_tasks.append(calling_task)

    def finish_prefetch(self, calling_task, task_list, device_list):
        """
        Finish prefetching and update data and device states.
        """
        for device in device_list:

            # if device.name in self.status_per_device:
            assert(device.name in self.status_per_device)
            self.status_per_device[device.name].in_progress = False

            self.status_per_device[device.name].moving_tasks.remove(calling_task)
            # finished_task = self.status_per_device[device.name].moving_tasks.get()[1]
            # assert(finished_task == calling_task)

            # print("FINISH Prefetch Check", self)

    def use(self, task, device_list, is_movement):
        """
        Decrement prefetch counter
        """
        for device in device_list:

            if device.name in self.status_per_device and not is_movement:
                self.status_per_device[device.name].prefetched -= 1
                # print("USEING")
                # print(self.status_per_device[device.name].dependent_tasks)
                # print(task.name)
                # TODO(hc): why doens't it increase moving task counter?
                self.status_per_device[device.name].dependent_tasks.remove(task)
            elif not is_movement:
                raise Exception(
                    f"Invalid State. Data is not available by task runtime. \n\t {task} | \n\t {data} | \n\t {device}")

    def read(self, task, device_list, is_movement=False):
        """
        Decrement prefetch counter
        """
        self.use(task, device_list, is_movement)

    def write(self, task, device_list, is_movement=False):
        """
        A task writes this PArray, and so, invalidate other copies on other
        different devices.
        """
        # NOTE: Write data isn't prefetched.
        # Decrease prefetch count
        # self.use(task, device_list, is_movement)

        #Check if data is already being used on device.

        if not is_movement:

            for device in device_list:
                if(self.status_per_device[device.name].used > 1):
                    print(self.name, self.status_per_device)
                    assert(False)

            for device_name in self.status_per_device.keys():
                if SyntheticDevice.devicespace[device_name] not in device_list:
                    if(self.status_per_device[device_name].used > 0):
                        print(self.name, self.status_per_device)
                        assert(False)

            # Mark stale copies of this device on other devices.
            device_name_list = [d.name for d in device_list]
            for device_name in self.status_per_device.keys():
                if device_name not in device_name_list:
                    self.status_per_device[device_name].stale += 1

            # Clear stale data from devices
            # NOTE: Should be no-op if is_movement is True
            self.evict_stale()

    def active_transfer(self, device):
        """
        True if this PArray currently is being moved/created.
        """
        if status := self.get_status(device):
            return status.in_progress
        else:
            return False

    def active_use(self, device):
        """
        True if this PArray currently is used by a task.
        """
        if status := self.get_status(device):
            return status.used > 0
        return False

    def valid_and_need(self, device):
        """
        True if this data is already prefetched and needed for any other
        scheduled tasks.
        """
        if status := self.get_status(device):
            # TODO(hc): how is prefetched used?
            # use() and read() on a device decreases that device's prefetch
            # counter. But use() and read() mean the valid data is on the
            # device. Then why is this valid?
            # shouldn't it be stale?
            return status.prefetched > 0
        return False

    def get_valid_sources(self, allow_in_progress=False):
        """
        Return a list of devices that have valid copies of this PArray.
        """
        valid_sources = []
        for device_name in self.status_per_device.keys():
            device = SyntheticDevice.devicespace[device_name]
            if self.valid(device, allow_in_progress=allow_in_progress):
                valid_sources.append(device)
        return valid_sources

    def choose_source_device(self, target_device, pool, required=False):
        """
        Choose a source device having the valid copy and that would be used
        to copy a data to another device.
        """
        topology = pool.topology

        # if already in progress at target, return that as the source
        # print("CHECK SOURCES IN PROGRESS :: ", self, target_device)
        if self.active_transfer(target_device):
            return target_device

        # otherwise, find the best source

        # Get list of non-stale devices that hold this data
        source_list = self.get_valid_sources()
        # print("SOURCE LIST", source_list)

        # Find the closest free source for the data
        closest_free_source = topology.find_best(
            target_device, source_list, free=True)

        return closest_free_source

    def __str__(self):
        return f"Data: {self.name} {self.size} | State: {self.status_per_device}"

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return str(self.name)

    def __eq__(self, other):
        return self.name == other.name

