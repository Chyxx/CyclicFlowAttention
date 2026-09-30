# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Derived from Flash Linear Attention and licensed under the MIT license
# found in the LICENSE file in this directory.
# For a list of upstream contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from dataclasses import dataclass, field

import transformers
from transformers import HfArgumentParser
from transformers import TrainingArguments as HFTrainingArguments


@dataclass
class TrainingArguments(HFTrainingArguments):
    model_name_or_path: str = field(
        default=None,
        metadata={"help": "Path to a CyclicFlowAttention config or checkpoint."},
    )
    tokenizer: str = field(
        default="fla-hub/gla-1.3B-100B",
        metadata={"help": "Tokenizer path or Hugging Face identifier."},
    )
    use_fast_tokenizer: bool = field(default=True)
    from_config: bool = field(
        default=True,
        metadata={"help": "Initialize from the config instead of loading model weights."},
    )
    cache_dir: str = field(
        default=None,
        metadata={"help": "Path to a tokenized Hugging Face Dataset saved to disk."},
    )
    split: str = field(default="train")
    varlen: bool = field(default=False)


def get_train_args() -> TrainingArguments:
    parser = HfArgumentParser(TrainingArguments)
    args, unknown_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    if unknown_args:
        raise ValueError(f"Unknown training arguments: {unknown_args}")
    if args.should_log:
        transformers.utils.logging.set_verbosity(args.get_process_log_level())
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    transformers.set_seed(args.seed)
    return args
