#!/bin/bash

while getopts "w:" opt; do
  case ${opt} in
    w )
      ENV_WORLD_SIZE=$OPTARG
      echo "ENV_WORLD_SIZE: $ENV_WORLD_SIZE"
      ;;
    \? )
      echo "잘못된 옵션: -$OPTARG" 1>&2
      ;;
  esac
done


LLAMA_8B_TOML_PATH="./llama3_8b.toml"

cat <<EOL > $LLAMA_8B_TOML_PATH
# torchtitan Config.toml
# NOTE: this toml config is a preset for 64 A100 GPUs.

[job]
dump_folder = "./outputs"
description = "Llama 3 8B training"

[profiling]
enable_profiling = true
save_traces_folder = "profile_trace"
profile_freq = 100

[metrics]
log_freq = 10
enable_tensorboard = true
save_tb_folder = "tb"

[model]
name = "llama3"
flavor = "8B"
norm_type = "rmsnorm"  # layernorm / np_layernorm / rmsnorm
tokenizer_path = "./assets/tokenizer/original/tokenizer.model"
# converters = "float8"
enable_finetune = true
model_path = "./assets/tokenizer/original/"
model_files = ['model-00001-of-00004.safetensors', 'model-00002-of-00004.safetensors', 'model-00003-of-00004.safetensors', 'model-00004-of-00004.safetensors']


[optimizer]
name = "AdamW"
lr = 3e-4
eps = 1e-8

[lr_scheduler]
warmup_steps = 200  # lr scheduler warm up

[training]
batch_size = $ENV_WORLD_SIZE
seq_len = 8192
max_norm = 1.0  # grad norm clipping
steps = 1000
data_parallel_replicate_degree = 1
data_parallel_shard_degree = -1
tensor_parallel_degree = 1
compile = false
dataset = "c4_test"

[experimental]
context_parallel_degree = 1
pipeline_parallel_degree = $ENV_WORLD_SIZE

[checkpoint]
enable_checkpoint = false
folder = "checkpoint"
interval = 500
model_weights_only = false
export_dtype = "float32"
async_mode = "disabled" # ["disabled", "async", "async_with_pinned_mem"]

[activation_checkpoint]
mode = 'selective'  # ['none', 'selective', 'full']
selective_ac_option = 'op'  # 'int' = ac every positive int layer or 'op', ac based on ops policy

[float8]
enable_fsdp_float8_all_gather = false
precompute_float8_dynamic_scale_for_fsdp = false
EOL



