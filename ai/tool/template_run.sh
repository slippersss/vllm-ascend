rm -rf /root/ascend/log/debug/*
export OMP_NUM_THREADS=10
export OMP_PROC_BIND=false
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_USE_V1=1
export VLLM_RPC_TIMEOUT=3600000
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=30000


MODEL_PATH=xxx
HOST_IP=xxx
ARGS=(
    --model $MODEL_PATH
    --served-model-name foobar
    --host $HOST_IP
    --port 60006
    --data-parallel-size 2
    --tensor-parallel-size 4
    --enable-expert-parallel
    --gpu-memory-utilization 0.95
    --max-model-len 32768
    --max-num-batched-tokens 32768
    --max-num-seqs 32
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": [4, 8, 16, 32, 64, 128]}'
    # --enforce-eager
    # --no-async-scheduling
    # --no-enable-prefix-caching
    # --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/l00957369/data", "torch_profiler_with_stack": true}'
)
vllm serve "${ARGS[@]}"
exit


curl http://xxx:60006/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "foobar",
        "prompt": "San Francisco is a",
        "max_tokens": 5,
        "temperature": 0
    }'
curl -X POST http://xxx:60006/start_profile
curl -X POST http://xxx:60006/stop_profile
ais_bench --models vllm_api_stream_chat --datasets gsm8k_gen_0_shot_cot_chat_prompt --debug --num-warmups 0 --dump-eval-details
