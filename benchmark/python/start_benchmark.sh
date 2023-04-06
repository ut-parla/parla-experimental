#!/bin/bash

SOURCE=${BASH_SOURCE[0]}
DIR="$( dirname "${SOURCE}" )"

#PLACEMENT_MODE=( "fixed-placement" "policy" )
PLACEMENT_MODE=( "policy" )

#DATA_MOVE_MODE=( "no" "lazy" "eager" )
DATA_MOVE_MODE=( "eager" )


NUM_GPUS=4
for d_mode in "${DATA_MOVE_MODE[@]}"; do
  for ps in "${PLACEMENT_MODE[@]}"; do
    for ((i=0; i<NUM_GPUS; ++i)); do
      CUDA_VISIBLE_DEVICES=""
      for ((j=0; j<=i; ++j)); do
          CUDA_VISIBLE_DEVICES+=$j
          if [ $j -lt $i ]
          then
              CUDA_VISIBLE_DEVICES+=","
          fi
      done
      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${DIR}/benchmark.py $((i+1)) $ps $d_mode
      echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ${DIR}/benchmark.py $((i+1)) $ps $d_mode"
    done
  done
done
