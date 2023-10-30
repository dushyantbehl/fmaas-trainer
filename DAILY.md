## Daily notes on progress (reverse chrono order)

### 10/30

### 10/27
* First run on Llama 13B with an instruction dataset gave ~900 tokens/sec/GPU (run is logged in Wandb (here)[https://wandb.ai/wx-tuning/huggingface/runs/bzx3fntf])

### 10/26
* Have the first working version of tuning based on (Stanford Alpaca)[https://github.com/tatsu-lab/stanford_alpaca]
* (Code location)[https://github.ibm.com/rganti/fmaas-tuning/]
* First run using Flash attention v2 with Llama
* GPTBigCode in `transformers` today does not have Flash V2, expected to be available soon (see this Slack (thread)[https://ibm-research.slack.com/archives/C0576DSN62U/p1698311316290499?thread_ts=1698268904.399859&cid=C0576DSN62U] with HF)
