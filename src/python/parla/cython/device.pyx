from parla.cython import core


cdef class DeviceBinder:
    cdef Device* device_

    cpdef get_id(self):
        return self.device_.GetID()

    cpdef get_name(self):
        return self.device_.GetName().decode()

    def __dealloc__(self):
        del self.device_


cdef class CUDADeviceBinder(DeviceBinder):
    def __cinit__(self, int dev_id):
        self.device_ = new CUDADevice(dev_id)

    def __init__(self, dev_id: int):
        pass

    def __repr__(self):
        return f"[Device] {self.get_name()}"


cdef class CPUDeviceBinder(DeviceBinder):
    def __cinit__(self, int dev_id):
        self.device_ = new CPUDevice(dev_id)

    def __init__(self, dev_id: int):
        pass

    def __repr__(self):
        return f"[Device] {self.get_name()}"


cdef class CyDeviceManager:
    """
    This class manages devices on the current system.
    For convenience, it registers devices specified by users to both
    pure Python and Cython side; So, it replicates processes.
    """
    cdef DeviceManager* device_manager_
    cdef public int num_cpus_
    cdef public int num_gpus_
    cdef public list pydevice_manager_

    def __cinit__(self):
        # Set the number of cpus and gpus.
        # This is not accessing OS to get HW topology information,
        # but rely on a developer's knowledge.
        self.num_cpus_ = 1
        self.num_gpus_ = 4

        self.device_manager_ = new DeviceManager(\
                                    self.num_cpus_, self.num_gpus_)
        self.device_manager_.RegisterDevices()

    def __init__(self):
        self.pydevice_manager_ = []
        self.register_devices()

    def __dealloc__(self):
        del self.device_manager_

    def register_devices(self):
        print(type(self.pydevice_manager_))
        for d in range(self.num_cpus_):
            self.pydevice_manager_.append(CPUDeviceBinder(d))

        for d in range(self.num_gpus_):
            self.pydevice_manager_.append(CUDADeviceBinder(d))

    def get_all_devices(self):
        return self.pydevice_manager_

    cdef DeviceManager* get_cpp_device_manager_ptr(self):
        return self.device_manager_ 

    """
    TODO(hc): the return type should be a vector reference.
              first, is this a good pattern?
    cdef vector[Device] GetCyDevices(self):
        return self.device_manager_.GetAllDevices()
    """
