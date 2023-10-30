# FMaaS Tuning

This repo aims to create basic tuning scripts independent of other pieces, assumes the use of Hugging Face and PyTorch FSDP only. Our approach to tuning is:
1. Models are loaded from Hugging Face `transformers` -- which the current models are optimized for using `Flash Attention v2`
2. Hugging Face `Trainer` for the training loop
3. `FSDP` as the backend for training

The code is based on [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca)

## Installation

```
pip install -r requirements.txt
pip install -e .
```

## Training

```
torchrun --nproc_per_node=8
--master_port=1234 
train.py 
--model_name_or_path /lustre/llama_weights/hf/13B 
--data_path ./alpaca_data.json 
--bf16 True 
--output_dir ./alpaca-tuned-13B 
--num_train_epochs 3 
--per_device_train_batch_size 4 
--per_device_eval_batch_size 4 
--evaluation_strategy "no" 
--save_strategy "steps" 
--save_steps 2000 
--save_total_limit 1 
--learning_rate 2e-5 
--weight_decay 0. 
--warmup_ratio 0.03 
--lr_scheduler_type "cosine" 
--logging_steps 1 
--fsdp "full_shard auto_wrap" 
--fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer'
--include_tokens_per_second
```

The above is an example. We would need to tune parameters depending on the model size, data size. The above example has been validated on 8 x A100 80GB.
