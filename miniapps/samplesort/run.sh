echo "cupy only"
python sample_sort_cupy.py -r 10 -w 5 -n 500000000 -gpu 1 -check 0
python sample_sort_cupy.py -r 10 -w 5 -n 500000000 -gpu 2 -check 0
python sample_sort_cupy.py -r 10 -w 5 -n 500000000 -gpu 3 -check 0
python sample_sort_cupy.py -r 10 -w 5 -n 500000000 -gpu 4 -check 0

echo "crosspy w thread"
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 1 -check 0 -m crosspy
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 2 -check 0 -m crosspy
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 3 -check 0 -m crosspy
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 4 -check 0 -m crosspy

echo "crosspy w parla"
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 1 -check 0 -m parla
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 2 -check 0 -m parla
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 3 -check 0 -m parla
python sample_sort.py -r 10 -w 10 -n 500000000 -gpu 4 -check 0 -m parla


echo "cupy only"
python sample_sort_cupy.py -r 10 -w 10 -n 100000000 -gpu 1 -check 0
python sample_sort_cupy.py -r 10 -w 10 -n 200000000 -gpu 2 -check 0
python sample_sort_cupy.py -r 10 -w 10 -n 300000000 -gpu 3 -check 0
python sample_sort_cupy.py -r 10 -w 10 -n 400000000 -gpu 4 -check 0

echo "crosspy w thread"
python sample_sort.py -r 10 -w 10 -n 100000000 -gpu 1 -check 0 -m crosspy
python sample_sort.py -r 10 -w 10 -n 200000000 -gpu 2 -check 0 -m crosspy
python sample_sort.py -r 10 -w 10 -n 300000000 -gpu 3 -check 0 -m crosspy
python sample_sort.py -r 10 -w 10 -n 400000000 -gpu 4 -check 0 -m crosspy

echo "crosspy w parla"
python sample_sort.py -r 10 -w 10 -n 100000000 -gpu 1 -check 0 -m parla
python sample_sort.py -r 10 -w 10 -n 200000000 -gpu 2 -check 0 -m parla
python sample_sort.py -r 10 -w 10 -n 300000000 -gpu 3 -check 0 -m parla
python sample_sort.py -r 10 -w 10 -n 400000000 -gpu 4 -check 0 -m parla