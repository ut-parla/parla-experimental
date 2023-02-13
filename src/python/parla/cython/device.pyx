import cython

cdef class ArchitectureBinder:
    cdef Architecture* arch_

    def __cinit__(self, string arch_name, ArchID arch_id):
        self.arch_ = new Architecture(arch_name, arch_id)

    cpdef get_name(self):
        return self.arch_.GetName()

    cpdef get_id(self):
        return self.arch_.GetID()


cdef class CUDAArchBinder(ArchitectureBinder):
    cdef CUDAArch cuda_arch_
 

cdef class CPUArchBinder(ArchitectureBinder):
    cdef CPUArch cpu_arch_


cdef class DeviceBinder:
    cdef Device device_

    def __cinit__(self):
        pass

    def __init__(self):
        pass


cdef class CUDADeviceBinder(DeviceBinder):
    cdef CUDADevice cu_device_ 

    def __cinit__(self):
        pass

    def __init__(self):
        pass


cdef class CPUDeviceBinder(DeviceBinder):
    cdef CPUDevice cpu_device_ 

    def __cinit__(self):
        pass

    def __init__(self):
        pass
