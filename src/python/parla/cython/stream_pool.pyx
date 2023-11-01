

cdef class CyStreamPool:

    cdef InnerStreamPool* _c_pool
    cdef dict _pool
    cdef int _per_device
    cdef list _device_list

    def __cinit__(self):
        self._c_pool = new InnerStreamPool()

    def __dealloc__(self):
        del self._c_pool

    def __init__(self, device_list, per_device=8, cupy_flag=False):

        if cupy_flag:
            self.StreamClass = CupyStream 
        else:
            self.StreamClass = Stream

        self._device_list = device_list
        self._per_device = per_device
        self._pool = {}

        for device in self._device_list:
            self._pool[device] = []
            
            with device.device as d:
                for i in range(self._per_device):
                    self._pool[device].append(self.StreamClass(device=device))

    def get_stream(self, device):
        if len(self._pool[device]) == 0:
            #Create a new stream if the pool is empty.
            new_stream = self.StreamClass(device=device)
            return new_stream

        return self._pool[device].pop()

    def return_stream(self, stream):
        self._pool[stream.device].append(stream)

    def __summarize__(self):
        summary  = ""
        for device in self._device_list:
            summary += f"({device} : {len(self._pool[device])})"

        return summary

    def __repr__(self):
        return f"StreamPool({self.__summarize__()})"






class StreamPool:

    def __init__(self, device_list, per_device=8):

        if CUPY_ENABLED:
            self.StreamClass = CupyStream 
        else:
            self.StreamClass = Stream

        self._device_list = device_list
        self._per_device = per_device
        self._pool = {}

        for device in self._device_list:
            self._pool[device] = []
            
            with device.device as d:
                for i in range(self._per_device):
                    self._pool[device].append(self.StreamClass(device=device))

    def get_stream(self, device):
        if len(self._pool[device]) == 0:
            #Create a new stream if the pool is empty.
            new_stream = self.StreamClass(device=device)
            return new_stream

        return self._pool[device].pop()

    def return_stream(self, stream):
        self._pool[stream.device].append(stream)

    def __summarize__(self):
        summary  = ""
        for device in self._device_list:
            summary += f"({device} : {len(self._pool[device])})"

        return summary

    def __repr__(self):
        return f"StreamPool({self.__summarize__()})"

