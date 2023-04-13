import argparse
from parla import Parla
import tempfile
import os
import cholesky


parser = argparse.ArgumentParser()
parser.add_argument('-workers', type=int, default=1)
parser.add_argument('-perthread', type=int, default=1)
parser.add_argument('-matrix', default=None)
parser.add_argument('-t', type=int, default=1)
parser.add_argument('-process', type=int, default=0)
parser.add_argument('-b', type=int, default=2000)
args = parser.parse_args()

with Parla():
    print("Setup: ", args.b, args.workers)
    cholesky.run(matrix="chol_20000.npy", n=20000, block_size=args.b, check_error=False, workers=args.workers)

with Parla():
    cholesky.run(matrix="chol_20000.npy", n=20000, block_size=args.b, check_error=False, workers=args.workers)

with Parla():
    cholesky.run(matrix="chol_20000.npy", n=20000, block_size=args.b, check_error=False, workers=args.workers)
