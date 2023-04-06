GRAPH_TYPES_STR=( "serial" "independent" "reduction" )
#GRAPH_TYPES_STR=( "independent" )
#NUM_TASKS_SET=( 300 500 1000 2000 )
NUM_TASKS_SET=( 300 )
#LEVELS=( 8 16 )
LEVELS=( 8 )
#SLEEP_KNOBS=( 3000 5000 10000 16000 20000 )
SLEEP_KNOBS=( 16000 )
#FD_DATA_KNOBS=( 6250 62500 625000 6250000 )
FD_DATA_KNOBS=( 6250 )
#SD_DATA_KNOBS=( 1 2 )
SD_DATA_KNOBS=( 2 )
NUM_GPUS_SET=( "1" "2" "3" "4")
CUDA_VISIBLE_DEVICES_SET=( "0" "0,1" "0,1,2" "0,1,2,3" )
USER_CHOSEN_PLACEMENT_SET=( "0" "1" )
GIL_COUNT=1
GIL_TIME=0

DATA_MOVE_MODES=( 0 1 2 )

GRAPH_DIR="graphs"

GRAPH_INPUT_DIR="sc23_inputs"
#rm -rf $GRAPH_INPUT_DIR
#mkdir $GRAPH_INPUT_DIR

OUTPUT_DIR="sc23_outputs"
#rm -rf $OUTPUT_DIR
#mkdir $OUTPUT_DIR

SOURCE=${BASH_SOURCE[0]}
DIR="$( dirname "${SOURCE}" )"

for GRAPH_TYPE in "${GRAPH_TYPES_STR[@]}"; do
  for computation_time in "${SLEEP_KNOBS[@]}"; do
    for fd_data_knob in "${FD_DATA_KNOBS[@]}"; do
      for sd_data_knob in "${SD_DATA_KNOBS[@]}"; do 
        for user_chosen_placement in "${USER_CHOSEN_PLACEMENT_SET[@]}"; do
          if [[ ${GRAPH_TYPE} == *"reduction"* ]]; then
            for level in "${LEVELS[@]}"; do
              for num_gpus in "${!NUM_GPUS_SET[@]}"; do
                ng=$((num_gpus + 1))
                for data_move_mode in "${DATA_MOVE_MODES[@]}"; do
                  FLAGS=" -computation_weight "${computation_time}" -gil_count "$GIL_COUNT" -gil_time "$GIL_TIME" -user "$user_chosen_placement" -num_gpus "$ng
                  FLAGS+=" -overlap 1 -level "${level}" -branch 2 -d "${sd_data_knob}" -data_move "${data_move_mode}
                  FLAGS+=" -workloads reduction -width_bytes "${fd_data_knob}
                  output_prefix="${GRAPH_TYPE}_${fd_data_knob}_${sd_data_knob}_${computation_time}_${user_chosen_placement}_${level}_${ng}_${data_move_mode}"
                  if [[ ${fd_data_knob} == 0 ]]; then
                      output_prefix+=".nodata"
                  fi
                  output_fname=${output_prefix}.log
                  commands="python "${DIR}"/benchmark.py -graph ${GRAPH_INPUT_DIR}/${output_prefix}.gph "$FLAGS
                  echo $commands
#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$num_gpus]} $commands > $OUTPUT_DIR/${output_fname}
                  grep "reduction," ${OUTPUT_DIR}/${output_fname} >> ${OUTPUT_DIR}/result.out
                done
              done
            done
          else
            for num_task in "${NUM_TASKS_SET[@]}"; do
              for num_gpus in "${!NUM_GPUS_SET[@]}"; do
                ng=$((num_gpus + 1))
                for data_move_mode in "${DATA_MOVE_MODES[@]}"; do
                  FLAGS=" -computation_weight "${computation_time}" -gil_count "$GIL_COUNT" -gil_time "$GIL_TIME" -user "$user_chosen_placement" -num_gpus "$ng
                  FLAGS+=" -d "${sd_data_knob}" -data_move "${data_move_mode}
                  FLAGS+=" -width_bytes "${fd_data_knob}
                  if [[ ${GRAPH_TYPE} == *"independent"* ]]; then
                    FLAGS+=" -overlap 0 -n "${num_task}" -workloads independent"
                  elif [[ ${GRAPH_TYPE} == *"serial"* ]]; then
                    FLAGS+=" -overlap 1 -n "${num_task}" -workloads serial"
                  fi
                  output_prefix="${GRAPH_TYPE}_${fd_data_knob}_${sd_data_knob}_${computation_time}_${user_chosen_placement}_${num_task}_${ng}_${data_move_mode}"
                  if [[ ${fd_data_knob} == 0 ]]; then
                      output_prefix+=".nodata"
                  fi
                  commands="python "${DIR}"/benchmark.py -graph ${GRAPH_INPUT_DIR}/${output_prefix}.gph "$FLAGS
                  output_fname=${output_prefix}.log
                  echo $output_prefix
                  echo $commands
#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$num_gpus]} $commands > $OUTPUT_DIR/${output_fname}
                  if [[ ${GRAPH_TYPE} == *"independent"* ]]; then
                      grep "independent," ${OUTPUT_DIR}/${output_fname} >> ${OUTPUT_DIR}/result.out
                  elif [[ ${GRAPH_TYPE} == *"serial"* ]]; then
                      grep "serial," ${OUTPUT_DIR}/${output_fname} >> ${OUTPUT_DIR}/result.out
                  fi
                done
              done
            done
          fi
        done
      done
    done
  done
done
