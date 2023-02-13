from parla import Parla
from parla.cython import device 

CUDADeviceBinder = device.CUDADeviceBinder

def test_device():
    cuda_dev = CUDADeviceBinder(0) 
    print(cuda_dev)

if __name__ == "__main__":
    with Parla():
        test_device()
