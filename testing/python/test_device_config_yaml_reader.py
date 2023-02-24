import yaml
import pytest


def test_read_yaml():
    # TODO(hc): It does not contain pytests. I am not sure this is on
    #           the proper directory.
    # TODO(hc): It would be great if we have validation codes
    #           for YAML.
    file_path = "devices_sample.YAML"
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
        cpu_num_cores = config["CPU"]["num_cores"]
        cpu_mem_sz = config["CPU"]["mem_sz"]
        gpu_num_devices = config["GPU"]["num_devices"]
        gpu_mem_szs = config["GPU"]["mem_sz"]
        assert len(gpu_mem_szs) == gpu_num_devices
        assert cpu_num_cores == 48
        assert cpu_mem_sz == 180000000
        for mem_sz in gpu_mem_szs:
            assert mem_sz == 11000000
        print(f"CPU # of Cores: {cpu_num_cores}, memory size: {cpu_mem_sz},")
        print(
            f"GPU # of Devices: {gpu_num_devices}, memory sizes: {gpu_mem_szs}")
