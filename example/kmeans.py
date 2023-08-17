import numpy as np
#import random
import math

def main():
    numPoints = 100
    x = np.random.rand(numPoints,2)

    ks = 10
    kmeans = np.random.rand(ks,2)

    iter = 5    
    for iter in range(5):   # Iterate for iter times
        newPoints = np.zeros((10,2)) # Initialize arrays to zero
        count = np.zeros((10,1))
        for i in range(numPoints):
            min_j = 0
            prev = 1000
            for j in range(ks):
                curr = math.dist(x[i], kmeans[j])
                if(curr<prev):
                    min_j = j
                    prev = curr
            newPoints[min_j] += x[i]
            count[min_j] += 1
        kmeans = newPoints / count
        print(kmeans)
    #    means = np.empty((k,2))
    #    for i in range k:
    #        means[i] = np.random.random()


if __name__ == "__main__":
    main()