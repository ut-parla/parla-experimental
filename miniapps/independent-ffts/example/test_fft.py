import cupy as cp
import crosspy as xp
import numpy as np
from cufftmg import handler 


N = 40

x = np.arange(N**2, dtype=np.float64) * 2 + 1
y = (np.arange(N**2, dtype=np.float64) + 1) * 2

x = np.reshape(x, (N, N))
y = np.reshape(y, (N, N))
x = x + 1j * y

device_ids = [0, 1]
stream_list = []
cupy_list = []

step = N // len(device_ids)

for i in device_ids:
    with cp.cuda.Device(i):
        local_copy = x[i*step:(i+1)*step, :]
        local_copy = cp.asarray(local_copy)
        stream_list.append(cp.cuda.get_current_stream())
        cupy_list.append(local_copy)

array = xp.array(cupy_list, axis=0)


for arr in array.device_view():
    arr = arr[0]
    print(type(arr))
    print(arr.device)
    print(arr.shape)
    print(arr.data.mem.ptr)



h = handler()
stream = cp.cuda.get_current_stream()
h.configure(device_ids=device_ids, streams=stream_list, size=40)
h.fft2(array)
#print(array)
h.ifft2(array)

print(array)
