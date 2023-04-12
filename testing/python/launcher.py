import os
import sys
import argparse
import pexpect as pe
import re

def wassert(output, condition, required=True, verbose=True):
    if condition:
        return True
    else:
        print("\t   FAILURE:", output[1])
        if verbose:
            print("\t   OUTPUT:", output[0])

        if required:
            raise Exception("Assertion Failure.")
        else:
            return False

def parse_times(output):
    times = []
    for line in output.splitlines():
        line = str(line).strip('\'')
        if "Time" in line:
            times.append(float(line.split()[-1].strip()))
    return times

def run():
    """
    Figure 12. Dask. Run blocked cholesky on the CPU.
    """
    timeout=1000

    worker_list = [ 1, 2, 4 , 8, 16]
    for wi in range(len(worker_list)):
        n_workers = worker_list[wi]
        for b in [2000, 1000, 500, 250]:
            print("Setup: ", n_workers, b)
            command = f"python cholesky_driver.py -workers {n_workers} -b {b}"
            output = pe.run(command, timeout=timeout, withexitstatus=True)
            #Make sure no errors or timeout were thrown
            wassert(output, output[1] == 0)
            #Parse output
            times = parse_times(output[0])
            n_threads = n_workers
            print(f"\t\t    {n_threads} CPU cores: {times}")
    return output_dict

run()
