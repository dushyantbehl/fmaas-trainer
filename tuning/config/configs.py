from dataclasses import dataclass, field
from typing import Dict, Optional
import transformers

DEFAULT_CONTEXT_LENGTH=4096
DEFAULT_OPTIMIZER="adamw_torch"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data in JSONL format."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default=DEFAULT_OPTIMIZER)
    model_max_length: int = field(
        default=DEFAULT_CONTEXT_LENGTH,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
