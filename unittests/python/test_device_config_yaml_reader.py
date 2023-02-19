import yaml

def read_yaml(file_path):
    # TODO(hc): It does not contain pytests. I am not sure this is on
    #           the proper directory.
    # TODO(hc): It would be great if we have validation codes
    #           for YAML.
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)
        cpu_num_vcus = config["CPU"]["num_cores"]
        cpu_mem_sz = config["CPU"]["mem_sz"]
        gpu_num_vcus = config["GPU"]["num_devices"]
        gpu_mem_szs = config["GPU"]["mem_sz"]
        assert(len(gpu_mem_szs) == gpu_num_vcus)
        print(f"CPU # of VCUs: {cpu_num_vcus}, memory size: {cpu_mem_sz},")
        print(f"GPU # of VCUs: {gpu_num_vcus}, memory sizes: {gpu_mem_szs}")


read_yaml("devices_sample.YAML")
