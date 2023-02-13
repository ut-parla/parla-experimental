from parla.cython import core


cdef class DeviceBinder:
    cdef Device* device_

    cpdef get_id(self):
        return self.device_.GetID()

    cpdef get_name(self):
        return self.device_.GetName().decode()


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
