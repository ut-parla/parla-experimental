import cython


cdef class DeviceBinder:
    def __init__(self):
        pass

    cpdef get_name(self):
        return self.device_.GetName()

    cpdef get_id(self):
        return self.device_.GetID()


cdef class CUDADeviceBinder(DeviceBinder):
    cdef CUDADevice* device_

    def __cinit__(self, string dev_name, DevID dev_id):
        self.device_ = new CUDADevice(dev_name, dev_id)


cdef class CPUDeviceBinder(DeviceBinder):
    cdef CPUDevice* device_

    def __cinit__(self, string dev_name, DevID dev_id):
        self.device_ = new CPUDevice(dev_name, dev_id)
