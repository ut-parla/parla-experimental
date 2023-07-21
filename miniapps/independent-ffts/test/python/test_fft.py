import cupy as cp
import numpy as np
import crosspy as xp

from cufftmg import import_test


def test_fft():
    N = 4

    import_test()

    x = np.arange(N**2, dtype=np.float64) * 2 + 1
    y = (np.arange(N**2, dtype=np.float64) + 1) * 2

    x = np.reshape(x, (N, N))
    y = np.reshape(y, (N, N))
    x + 1j * y

    device_ids = [0, 1]
    cupy_list = []

    step = N // len(device_ids)

    for idx in device_ids:
        with cp.cuda.Device(idx):
            local_copy = x[i*step:(i+1)*step, :]
            local_copy = cp.asarray(local_copy)
            cupy_list.append(local_copy)

    array = xp.array(cupy_list, axis=0)

    print(array)

test_fft()

            
    
