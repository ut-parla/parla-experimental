#GRAPH_TYPES_STR=( "serial" "independent" "reduction" )
GRAPH_TYPES_STR=( "independent" )
#GRAPH_TYPES_STR=( "serial" )
#GRAPH_TYPES_STR=( "independent" "serial" )
#NUM_TASKS_SET=( 300 500 1000 2000 )
#NUM_TASKS_SET=( 1000 )
NUM_TASKS_SET=( 500 1000 2000 )
#NUM_TASKS_SET=( 2000 )
#NUM_TASKS_SET=( 10 )
#LEVELS=( 8 16 )
LEVELS=( 8 )
#SLEEP_KNOBS=( 3000 5000 10000 16000 20000 )
#SLEEP_KNOBS=( 500 1000 2000 4000 8000 16000 32000 64000 )
SLEEP_KNOBS=(  1000 )
#SLEEP_KNOBS=( 16000 32000 64000 )
#SLEEP_KNOBS=( 64000 )
#SLEEP_KNOBS=( 16000 )
#FD_DATA_KNOBS=( 6250 62500 625000 6250000 )
#FD_DATA_KNOBS=( 6250 )
#FD_DATA_KNOBS=( 12500 )
FD_DATA_KNOBS=( 0 )
#SD_DATA_KNOBS=( 1 2 )
SD_DATA_KNOBS=( 2 )
#NUM_GPUS_SET=( "1" "2" "3" "4")
NUM_GPUS_SET=( "3" "4" )
#NUM_GPUS_SET=("1")
#CUDA_VISIBLE_DEVICES_SET=( "0" "0,1" "0,1,2" "0,1,2,3" )
CUDA_VISIBLE_DEVICES_SET=( "0,1,2" "0,1,2,3" )
#CUDA_VISIBLE_DEVICES_SET=( "0" )
USER_CHOSEN_PLACEMENT_SET=( "0" "1" )
#USER_CHOSEN_PLACEMENT_SET=( "0" )
GIL_COUNT=1
GIL_TIME=0

#OUT_ITERS=( "1" "2" "3" )
OUT_ITERS=( "1" )

#DATA_MOVE_MODES=( 0 1 2 )
#DATA_MOVE_MODES=( 1 2 )
DATA_MOVE_MODES=( 0 )
#DATA_MOVE_MODES=( 2 )

BASE_DIR=04142023

#GRAPH_INPUT_DIR="asplos24_nodata_input_"
#rm -rf $GRAPH_INPUT_DIR
#mkdir $GRAPH_INPUT_DIR

#OUTPUT_DIR="asplos24_nodata_output_"
#rm -rf $OUTPUT_DIR
#mkdir $OUTPUT_DIR

SOURCE=${BASH_SOURCE[0]}
DIR="$( dirname "${SOURCE}" )"

for out_iter in "${OUT_ITERS[@]}"; do
  for GRAPH_TYPE in "${GRAPH_TYPES_STR[@]}"; do
    for computation_time in "${SLEEP_KNOBS[@]}"; do
      GRAPH_INPUT_DIR=$BASE_DIR"/asplos24_100KB_input_"$computation_time
      OUTPUT_DIR=$BASE_DIR"/asplos24_100KB_output_"$computation_time
      mkdir -p $GRAPH_INPUT_DIR
      mkdir -p $OUTPUT_DIR
      for fd_data_knob in "${FD_DATA_KNOBS[@]}"; do
        for sd_data_knob in "${SD_DATA_KNOBS[@]}"; do 
          for user_chosen_placement in "${USER_CHOSEN_PLACEMENT_SET[@]}"; do
            if [[ ${GRAPH_TYPE} == *"reduction"* ]]; then
              for level in "${LEVELS[@]}"; do
                for num_gpus in "${NUM_GPUS_SET[@]}"; do
                  for data_move_mode in "${DATA_MOVE_MODES[@]}"; do
                    FLAGS=" -computation_weight "${computation_time}" -gil_count "$GIL_COUNT" -gil_time "$GIL_TIME" -user "$user_chosen_placement" -num_gpus "$num_gpus
                    FLAGS+=" -overlap 1 -level "${level}" -branch 2 -d "${sd_data_knob}" -data_move "${data_move_mode}
                    FLAGS+=" -workloads reduction -width_bytes "${fd_data_knob}" -iter "${out_iter}
                    output_prefix="${GRAPH_TYPE}_${fd_data_knob}_${sd_data_knob}_${computation_time}_${user_chosen_placement}_${level}_${num_gpus}_${data_move_mode}_${out_iter}"
                    if [[ ${fd_data_knob} == 0 ]]; then
                        output_prefix+=".nodata"
                    fi
                    output_fname=${output_prefix}.log
                    commands="python "${DIR}"/benchmark.py -graph ${GRAPH_INPUT_DIR}/${output_prefix}.gph "$FLAGS
                    echo $commands
                    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$((num_gpus - 1))]} $commands"
                    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$((num_gpus - 1))]} $commands > $OUTPUT_DIR/${output_fname}
                    grep "reduction," ${OUTPUT_DIR}/${output_fname} >> ${OUTPUT_DIR}/result.out
                    grep "reduction," ${OUTPUT_DIR}/${output_fname} >> ${BASE_DIR}/result.out
                    sleep 5
                  done
                done
              done
            else
              for num_task in "${NUM_TASKS_SET[@]}"; do
                for ng_idx in "${!NUM_GPUS_SET[@]}"; do
                  for data_move_mode in "${DATA_MOVE_MODES[@]}"; do
                    num_gpus=${NUM_GPUS_SET[$ng_idx]}
                    FLAGS=" -computation_weight "${computation_time}" -gil_count "$GIL_COUNT" -gil_time "$GIL_TIME" -user "$user_chosen_placement" -num_gpus "$num_gpus
                    FLAGS+=" -d "${sd_data_knob}" -data_move "${data_move_mode}
                    FLAGS+=" -width_bytes "${fd_data_knob}" -iter "${out_iter}
                    if [[ ${GRAPH_TYPE} == *"independent"* ]]; then
                      FLAGS+=" -overlap 0 -n "${num_task}" -workloads independent"
                    elif [[ ${GRAPH_TYPE} == *"serial"* ]]; then
                      FLAGS+=" -overlap 1 -n "${num_task}" -workloads serial"
                    fi
                    output_prefix="${GRAPH_TYPE}_${fd_data_knob}_${sd_data_knob}_${computation_time}_${user_chosen_placement}_${num_task}_${num_gpus}_${data_move_mode}_${out_iter}"
                    if [[ ${fd_data_knob} == 0 ]]; then
                        output_prefix+=".nodata"
                    fi
                    commands="python -X faulthandler "${DIR}"/benchmark.py -graph ${GRAPH_INPUT_DIR}/${output_prefix}.gph "$FLAGS
                    output_fname=${output_prefix}.log
                    echo $output_prefix
                    echo $commands
#CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$num_gpus]} $commands > $OUTPUT_DIR/${output_fname}
                    echo "ng_idx:"$ng_idx
                    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$ng_idx]} $commands"
                    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_SET[$ng_idx]} $commands
                    if [[ ${GRAPH_TYPE} == *"independent"* ]]; then
                        grep "independent," ${OUTPUT_DIR}/${output_fname} >> ${OUTPUT_DIR}/result.out
                        grep "independent," ${OUTPUT_DIR}/${output_fname} >> ${BASE_DIR}/result.out
                    elif [[ ${GRAPH_TYPE} == *"serial"* ]]; then
                        grep "serial," ${OUTPUT_DIR}/${output_fname} >> ${OUTPUT_DIR}/result.out
                        grep "serial," ${OUTPUT_DIR}/${output_fname} >> ${BASE_DIR}/result.out
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
done
