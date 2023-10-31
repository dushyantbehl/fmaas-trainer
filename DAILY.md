## Daily notes on progress (reverse chrono order)

### 10/30
* HF suggests to use `SFTTrainer`, discussions with Gaurav et al underway
* A first run with `SFTTrainer` is now checked in, throughput appears to meet what we expect in 13B model from pretraining (~1600 toks/sec/GPU)
* To be investigated:
```
/home/rganti/anaconda3/envs/tuning/lib/python3.10/site-packages/trl/trainer/utils.py:548: UserWarning: The passed formatting_func has more than one argument. Usually that function should have a single argument `example` which corresponds to the dictionary returned by each element of the dataset. Make sure you know what you are doing.
  warnings.warn(
Token indices sequence length is longer than the specified maximum sequence length for this model (4206 > 4096). Running this sequence through the model will result in indexing errors
```

### 10/27
* First run on Llama 13B with an instruction dataset gave ~900 tokens/sec/GPU (run is logged in Wandb (here)[https://wandb.ai/wx-tuning/huggingface/runs/bzx3fntf])

### 10/26
* Have the first working version of tuning based on (Stanford Alpaca)[https://github.com/tatsu-lab/stanford_alpaca]
* (Code location)[https://github.ibm.com/rganti/fmaas-tuning/]
* First run using Flash attention v2 with Llama
* GPTBigCode in `transformers` today does not have Flash V2, expected to be available soon (see this Slack (thread)[https://ibm-research.slack.com/archives/C0576DSN62U/p1698311316290499?thread_ts=1698268904.399859&cid=C0576DSN62U] with HF)
