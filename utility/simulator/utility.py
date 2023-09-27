
units = {"B": 1, "KB": 10**3, "MB": 10**6, "GB": 10**9, "TB": 10**12} 

def parse_size(size_str: str):
    number, unit = [string.strip() for string in size_str.split()]
    return int(float(number) * units[unit])
