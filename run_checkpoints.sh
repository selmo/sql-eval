#!/bin/zsh

model_names=("sqlcoder_8b_fullft_ds_008_llama3_mgn1_b1_0900_b2_0990")

# Loop over model names
for model_name in "${model_names[@]}"; do
  # list the folder names in /models/combined/${model_name}
  model_dir="/home/ubuntu/.cache/huggingface/hub/${model_name}"
  echo "Model directory: ${model_dir}"
  checkpoints=($(ls $model_dir))
  echo "Checkpoints: ${checkpoints}"
  # Loop over checkpoints
  for checkpoint in "${checkpoints[@]}"; do
    # skip if does not start with "checkpoint-"
    if [[ ! $checkpoint == checkpoint-* ]]; then
      continue
    fi
    model_path="${model_dir}/${checkpoint}"
    checkpoint_num=$(echo $checkpoint | cut -d'-' -f2)
    echo "Running model ${model_name} checkpoint ${checkpoint_num}"
    # first, get the API up
    python3 utils/api_server.py --model "$model_path" --tensor-parallel-size 1 --dtype float16 --max-model-len 8192 --gpu-memory-utilization 0.90 --block-size 16 --disable-log-requests --port 8080 &

    # run a loop to check if the http://localhost:8080/health endpoint returns a valid 200 result
    while true; do
      http_status=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health)
      if [ "$http_status" -eq 200 ]; then
        echo "API server is up and running"
        break
      else
        echo "Waiting for API server to be up..."
        sleep 1
      fi
    done

    # then run sql-eval
    python3 main.py \  
-db postgres \
-q "data/questions_gen_postgres.csv" "data/instruct_basic_postgres.csv" "data/instruct_advanced_postgres.csv" \
-o "results/${model_name}/c${checkpoint_num}_v1_api.csv" "results/${model_name}/c${checkpoint_num}_v1_basic.csv" "results/${model_name}/c${checkpoint_num}_v1_advanced.csv"
-g api \
-b 1 \
-f prompts/prompt.md \
--api_url "http://localhost:8080/generate" \
--api_type "vllm" \
-p 10 \
-c 0
    # finally, kill the api server
    pkill -9 -f utils/api_server.py
  done
done
