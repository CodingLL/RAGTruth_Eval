# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(base_model: str,
         peft_model: str,
         output_dir: str,
         safe_tensor: bool,
         quantization_4bit: bool=False, 
         quantization_8bit: bool=False):
        
    model = AutoModelForCausalLM.from_pretrained( 
        base_model,
        load_in_4bit=quantization_4bit,
        load_in_8bit=quantization_8bit,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp", 
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model
    )
        
    model = PeftModel.from_pretrained(
        model, 
        peft_model, 
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder="tmp",
    )

    model = model.merge_and_unload()
    model.save_pretrained(output_dir, safe_serialization=safe_tensor)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(main)