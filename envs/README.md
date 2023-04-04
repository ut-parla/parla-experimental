```
#create env from frontera.yml
conda install gcc=10 gxx=10
conda deactivate
module purge
module load cuda
conda activate <env name>

make clean; PARLA_ENABLE_CUDA=1 CC=gcc CXX=g++ make
```
