#!/bin/bash

# Check if script name was provided as a command-line argument
if [ $# -ne 2 ]
then
  echo "Usage: $0 <script_name>  $1 <argsfile_name>"
  exit 1
fi

# Get script name from command-line argument
script_name=$1
argsfile_name=$2

# Check if argument file exists
if [ ! -f $argsfile_name ]
then
  echo "Error: <$argsfile_name> file not found"
  exit 1
fi

# Loop over the argument sets in the file and launch the Python script with each one
while read args
do
    # Extract the number of GPUs from the input arguments
    #num_gpus=$(echo $args | sed 's/-/\n/g' | grep 'ngpus' | cut -d' ' -f2)
    # Set CUDA_VISIBLE_DEVICES based on the number of GPUs
    #export CUDA_VISIBLE_DEVICES=$(seq -s ',' 0 $(($num_gpus-1)))
    echo "Running script $script_name with arguments: $args"
    #, using GPUs: $CUDA_VISIBLE_DEVICES"
    # Save output to a file with a name based on the input arguments
    output_file_name=""
    for arg in $args
    do
	arg=$(echo $arg | sed 's/-/_/g')
        output_file_name="${output_file_name}_${arg}"
    done
    output_file_name="${output_file_name:1}.txt"
    output_file_log="${output_file_name:1}.qdrep"
    echo "Saving output to: $output_file_name"
    nsys profile -o $output_file_log python $script_name $args |tee $output_file_name
done < $argsfile_name
