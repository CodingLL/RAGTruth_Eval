#!/bin/bash
echo "start merging lora weights!"

CUDA_VISIBLE_DEVICES=1 \
python ./merge_lora_weights.py \
--base_model meta-llama/Llama-2-13b-hf \
--peft_model ./exp/llama2_13b_lora_0217 \
--output_dir ./exp/llama2_13b_lora_0217_merge \
--safe_tensor True

echo "merge done!"

