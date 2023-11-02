## Daily notes on progress (reverse chrono order)

### 11/01-11/02
* Kicked off training on two different datasets and they are going well without `packing`
* Packing has issues with the loss curve and has been raised with HF (see thread (here)[https://ibm-research.slack.com/archives/C0576DSN62U/p1698935125641059])
* Loss curves for no packing on API dataset matches what Pavan and team see
* @pavithra and @dushyant to look into getting numbers for various batch sizes
* @raghu to look into how to get `GPTBigCode` with FA2 (which is now merged and will be available in the main release).
* Issues with SFT Trainer: (a) Packing discrepancy (see this (thread)[https://ibm-research.slack.com/archives/C0576DSN62U/p1698935125641059]), (b) How data is formatted (see this (thread)[https://ibm-research.slack.com/archives/C0576DSN62U/p1698935416128899] and (issue)[https://github.com/huggingface/trl/issues/944]), (c) How to compute tokens/sec/GPU (see this (thread)[https://ibm-research.slack.com/archives/C0576DSN62U/p1698936778849009] and this (issue)[https://github.com/huggingface/transformers/issues/27027]).
* We need to have a SFT Trainer implementation in the main watsonx repo

### 10/31
* Model was trained with 1500 tokens/sec/GPU with 4k sequence length and packing (TODO: Find from HF on how the packing is being done)
* HF provided contacts (use Slack (channel)[https://ibm-research.slack.com/archives/C0576DSN62U]): @Younes Belkada for `SFTTrainer` and @Zach Mueller
  for FSDP integration
* CCC being explored (have the platform queue access)

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
