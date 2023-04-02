```
module load gcc/9.1
module load cuda

make clean; PARLA_ENABLE_CUDA=1 CC=gcc CXX=g++ make
```
