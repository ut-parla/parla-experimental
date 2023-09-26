import device

SyntheticDevice = device.SyntheticDevice

class ResourcePool:

    def __init__(self):
        self.pool = dict()

    def __str__(self):
        return f"ResourcePool: {self.pool}"

    def __repr__(self):
        return f"ResourcePool: {self.pool}"

    def __hash__(self):
        return hash(self.name)

    def add_resource(self, device, resource, amount):
        if device not in self.pool:
            self.pool[device] = dict()
        self.pool[device][resource] = amount

    def add_resources(self, device, resources):
        for resource, amount in resources.items():
            self.add_resource(device, resource, amount)
        return True

    def set_toplogy(self, topology):
        self.topology = topology

        for device in self.topology.devices:
            self.add_resources(device.name, device.resources)

            # Add backlink to device (in case we ever need to query it there)
            device.resource_pool = self

    def finalize(self):
        self.max_pool = copy.deepcopy(self.pool)
        self.max_connections = copy.deepcopy(self.connections)

    def check_device_resources(self, device, requested_resources):
        # print(device, requested_resources, self.pool)
        if device.name not in self.pool:
            return False
        for resource, amount in requested_resources.items():
            if resource not in self.pool[device.name]:
                return False
            if resource == "memory":
                # print("Checking memory")
                # print(self.pool[device.name][resource],
                #      device.parray_memory_usage, amount)
                if (self.pool[device.name][resource] - device.parray_memory_usage) < amount:
                    return False
            else:
                if self.pool[device.name][resource] < amount:
                    return False
        return True

    def check_resources(self, requested_resources):
        for device_name, resources in requested_resources.items():
            device = SyntheticDevice.devicespace[device_name]
            if not self.check_device_resources(device, resources):
                return False
        return True

    def reserve_resources_device(self, device, requested_resources):
        for resource, amount in requested_resources.items():
            self.pool[device][resource] -= amount
        return True

    def free_resources_device(self, device, requested_resources):
        for resource, amount in requested_resources.items():
            self.pool[device][resource] += amount
        return True

    def reserve_resources(self, requested_resources):
        for device, resources in requested_resources.items():
            self.reserve_resources_device(device, resources)
        return True

    def free_resources(self, requested_resources):
        for device, resources in requested_resources.items():
            self.free_resources_device(device, resources)
        return True

    def delete_device(self, device):
        del self.pool[device]
        return True

    def delete_resource_device(self, device, resource):
        if device in self.pool:
            if resource in self.pool[device]:
                del self.pool[device][resource]
        return True

    def delete_resources_device(self, device, resources):
        for resource in resources:
            self.delete_resource(self, device, resource)
        return True

    def delete_resource(self, resource):
        for device in self.pool.keys():
            self.delete_resource_device(device, resource)
        return True

    def delete_resources(self, resources):
        for device in self.pool.keys():
            self.delete_resources_device(device, resources)
        return True
