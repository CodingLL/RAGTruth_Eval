# RAGTruth_Eval

This project contains a hallucination detection pipeline based on RAGTruth.

## 1. LoRA Weights merge

merge_script.py is a script used to merge the LoRA model with a base model. Usage:

bash lora_merge.sh

## 2. Huggingface Text-Generation-Inference

The LLM inference is based on huggingface TGI. Usage:

bash TGI.sh

## 3. Inference based on Async

We use AsyncInferenceClient to inference asynchronously. Usage:

bash hall_inference.sh

## 4. Compute Case Level and Span Level metrics

The computation pipeline is given in compute_metrics.py. Usage:

bash compute_metrics.sh