from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import transformers
import torch
import datasets

from tuning.data import tokenizer_data_utils
from tuning.config import configs

from aim_loader import get_aimstack_callback

def format_prompt_fn(example):
    text = f"{example['input']} \n[ANS] {example['output']}"
    
    return text

def format_prompt_fn_no_pack(example):
    output = []
    for i in range(len(example['input'])):
        text = f"{example['input'][i]} \n[ANS] {example['output'][i]}"
        output.append(text)

    return output
    

def train():
    parser = transformers.HfArgumentParser((configs.ModelArguments, configs.DataArguments, configs.TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
        use_flash_attention_2=True,
    )

    model.gradient_checkpointing_enable()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = configs.DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = configs.DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = configs.DEFAULT_BOS_TOKEN
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = configs.DEFAULT_UNK_TOKEN

    tokenizer_data_utils.tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )
    
    # load the data by parsing JSON
    json_dataset = datasets.load_dataset('json', data_files=data_args.data_path)
    print(len(json_dataset['train']))
    
    #response_template="\n[ANS]"
    #data_collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    callback = get_aimstack_callback()

    if training_args.packing:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=json_dataset['train'],
            formatting_func=format_prompt_fn,
            packing=True,
            args=training_args,
            max_seq_length=4096,
            callbacks=[callback]
        )
    else:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=json_dataset['train'],
            formatting_func=format_prompt_fn_no_pack,
            args=training_args,
            max_seq_length=4096,
            callbacks=[callback]
        )
    
    trainer.train()
    
if __name__ == "__main__":
    train()
