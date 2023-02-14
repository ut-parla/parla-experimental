from parla import Parla
from parla.cython import device 

CUDADeviceBinder = device.CUDADeviceBinder
CyDeviceManager = device.CyDeviceManager

def test_device():
    cuda_dev = CUDADeviceBinder(0) 
    print(cuda_dev)

    cdm = CyDeviceManager()
    print("Get Python devices:", flush=True)
    print(cdm.get_all_devices())
    #print("Get c++ devices:", flush=True)
    #cdm.GetCyDevices()

if __name__ == "__main__":
    with Parla():
        test_device()
