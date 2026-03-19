#!/bin/bash
# NOTE : Run it from the ecmac folder #

export PYTHONPATH=$PYTHONPATH:./talent_benchmark:./src
export KERAS_BACKEND=jax
echo $PYTHONPATH
start_time=$(date +%s)



# Default values
dataset_path="talent_benchmark/data"
ood_dataset_path="talent_benchmark/data_ood"
rect_search_iters=300
k_ratio=0.9
num_of_repetitions=5
num_of_worsening_sets=20
use_knr=True

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    --ood_dataset_path)
      ood_dataset_path="$2"
      shift 2
      ;;
    --rect_search_iters)
      rect_search_iters="$2"
      shift 2
      ;;
    --k_ratio)
      k_ratio="$2"
      shift 2
      ;;
    --num_of_repetitions)
      num_of_repetitions="$2"
      shift 2
      ;;
    --num_of_worsening_sets)
      num_of_worsening_sets="$2"
      shift 2
      ;;
    --use_knr)
      use_knr="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --dataset_path PATH           The path to the original datasets (default: talent_benchmark/data)"
      echo "  --ood_dataset_path PATH       The path to the saved new ood data (default: talent_benchmark/data_ood/adversarial_based)"
      echo "  --rect_search_iters VALUE     The number of iterations the algorithm that breaks the data with run (default: 300)"
      echo "  --k_ratio VALUE               Ration of Train / Test Data (default: 0.9)"
      echo "  --num_of_repetitions VALUE    Number of times the data breaking algorithm will run again (default: 1)"
      echo "  --num_of_worsening_sets VALUE How many worsening epochs to keep (default: 7)"
      echo "  --use_knr VALUE               Use the KNR or the nuSVR algorithms to break the data (default: True)"
      echo "  --help                        Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done


MAX_JOBS=48 #48

# Get only subfolder names
subfolders=("$dataset_path"/*)
subfolders=("${subfolders[@]#"$dataset_path"/}")  # Remove dataset_path prefix
subfolders=("${subfolders[@]%/}")

total=${#subfolders[@]}
current=0

echo "Found $total subfolders to process from $dataset_path"

declare -a cpu_free
for ((i=0; i<MAX_JOBS; i++)); do
    cpu_free[$i]=1
done

# Map PID to CPU
declare -A pid_to_cpu

for subfolder in "${subfolders[@]}"; do
    # Wait for a free CPU
    while true; do
        free_cpu=-1
        for ((i=0; i<MAX_JOBS; i++)); do
            if [ ${cpu_free[$i]} -eq 1 ]; then
                free_cpu=$i
                break
            fi
        done

        if [ $free_cpu -ge 0 ]; then
            break
        fi

        # Check for finished jobs and free their CPUs
        for pid in "${!pid_to_cpu[@]}"; do
            if ! kill -0 $pid 2>/dev/null; then
                cpu=${pid_to_cpu[$pid]}
                cpu_free[$cpu]=1
                unset 'pid_to_cpu[$pid]'
                echo "Finished: PID $pid (freed CPU $cpu)"
            fi
        done

        sleep 0.1
    done

    # Mark CPU as busy and launch job
    cpu_free[$free_cpu]=0
    taskset -c $free_cpu python ./src/analysis/ood/adversarial_based/adv_based_ood_data_creator.py \
        --dataset "$subfolder" \
        --dataset_path "$dataset_path" \
        --ood_dataset_path "$ood_dataset_path" \
        --rect_search_iters "$rect_search_iters" \
        --k_ratio "$k_ratio" \
        --num_of_repetitions "$num_of_repetitions" \
        --num_of_worsening_sets "$num_of_worsening_sets" \
        --use_knr "$use_knr" &

    pid=$!
    pid_to_cpu[$pid]=$free_cpu
    echo "Started: $subfolder (CPU $free_cpu, PID $pid)"
done

# Wait for all jobs to complete
wait
echo "All $total subfolders processed!"


end_time=$(date +%s)

duration=$((end_time - start_time))
echo "Script took $duration seconds"