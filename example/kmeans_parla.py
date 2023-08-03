import numpy as np
import math
from parla import Parla
from parla.tasks import spawn, TaskSpace
from parla.devices import cpu, gpu

def main():
    @spawn(placement=cpu)
    async def main_task():
        numPoints = 100
        x = np.random.rand(numPoints,2)

        ks = 10
        kmeans = np.random.rand(ks,2)

        tasks = 10
        x_part = [x[i*10:(i+1)*10] for i in range(tasks)]
        k_part = [kmeans for i in range(tasks)]

        points_part = int(numPoints / tasks)    # Change to ciel

        T = TaskSpace('T')
        for i in range(tasks):
            @spawn(T[i], placement=cpu)
            def mytasks():
                newPoints = np.zeros((10,2))
                count = np.ones((10,1))
                for j in range(points_part):
                    min_dist = 0
                    prev = 1000
                    for k in range(ks):
                        curr = math.dist(x[j], kmeans[k])
                        if(curr<prev):
                            min_dist = k
                            prev = curr
                    newPoints[min_dist] += x[j]
                    count[min_dist] += 1                
                k_part = newPoints / count

        await T
        for i in range(tasks):
            kmeans += k_part[i]
        kmeans = kmeans/tasks



if __name__ == "__main__":
    with Parla():
        main()