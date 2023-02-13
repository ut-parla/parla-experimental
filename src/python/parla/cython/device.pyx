from parla.cython import core


cdef class DeviceBinder:
    def __init__(self):
        pass

    cpdef get_id(self):
        return self.device_.GetID()


cdef class CUDADeviceBinder(DeviceBinder):
    cdef CUDADevice* device_

    def __cinit__(self, int dev_id):
        self.device_ = new CUDADevice(dev_id)

    def __init__(self, dev_id: int):
        pass


cdef class CPUDeviceBinder(DeviceBinder):
    cdef CPUDevice* device_

    def __cinit__(self, int dev_id):
        self.device_ = new CPUDevice(dev_id)

    def __init__(self, dev_id: int):
        pass
