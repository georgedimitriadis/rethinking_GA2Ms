#!/bin/bash
# NOTE : Run it from the ecmac/talent_benchmark #

export PYTHONPATH=$PYTHONPATH:.:../src
export KERAS_BACKEND=jax
echo $PYTHONPATH 
start_time=$(date +%s)
dataset_path="./data"
logs_path="./logs"
results_model_path="./results_model"
tune_flag=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model_type)
      model_type="$2"
      shift 2
      ;;
    --dataset_path)
      dataset_path="$2"
      shift 2
      ;;
    --logs_path)
      logs_path="$2"
      shift 2
      ;;
    --results_model_path)
      results_model_path="$2"
      shift 2
      ;;
    --tune)
      tune_flag="--tune"
      shift 1
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo "Options:"
      echo "  --model_type VALUE          Model type (must be provided - no default)"
      echo "  --dataset_path PATH         Dataset path (default: ./data)"
      echo "  --logs_path PATH            Logs path (default: ./logs)"
      echo "  --results_model_path PATH   Results model path (default: ./results_model)"
      echo "  --tune                      Turn on the validation tuning of the ecmac linear layer"
      echo "  --help                      Show this help message"
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
#subfolders=(data/*/)
#subfolders=("${subfolders[@]#data/}")  # Remove 'data/' prefix
#subfolders=("${subfolders[@]%/}")      # Remove trailing '/'

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
                unset pid_to_cpu[$pid]
                echo "Finished: PID $pid (freed CPU $cpu)"
            fi
        done

        sleep 0.1
    done

    # Mark CPU as busy and launch job
    cpu_free[$free_cpu]=0
    taskset -c $free_cpu python ./training_calls/train_model_classical.py \
        --model_type "$model_type" \
        --dataset "$subfolder" \
        --dataset_path "$dataset_path" \
        --model_path "$results_model_path" \
        $tune_flag \
        > "./${logs_path}/${subfolder}_${model_type}.txt" \
        2> "./${logs_path}/${subfolder}_${model_type}.err" &

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
