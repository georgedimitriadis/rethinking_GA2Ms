#!/bin/bash
# NOTE : Quote it else use array to avoid problems #

export PYTHONPATH=$PYTHONPATH:.
start_time=$(date +%s)
model_type="$1"
parent_folder="./data"
for f in "$parent_folder"/*;
do
  folder_name=$(basename "$f")
  echo "Processing $folder_name ..."
  # take action on each file. $f store current file name
  python ./test/train_model_classical.py --model_type "$model_type" --dataset "$folder_name" > "./logs/${folder_name}_${model_type}.txt"
done

end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Script took $duration seconds"
