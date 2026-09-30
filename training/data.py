# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Derived from Flash Linear Attention and licensed under the MIT license
# found in the LICENSE file in this directory.
# For a list of upstream contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import PreTrainedTokenizer


@dataclass
class DataCollatorForLanguageModeling:
    tokenizer: PreTrainedTokenizer
    varlen: bool = False
    return_tensors: str = "pt"

    def __call__(
        self,
        examples: list[list[int] | dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(examples[0], dict):
            examples = [{"input_ids": example} for example in examples]

        def tensorize(example: dict[str, Any]) -> dict[str, Any]:
            tensorized = {}
            for key in ["input_ids", "offsets"]:
                if key not in example:
                    continue
                if isinstance(example[key], list):
                    tensorized[key] = torch.tensor(example[key], dtype=torch.long)
                elif isinstance(example[key], np.ndarray):
                    tensorized[key] = torch.from_numpy(example[key])
                else:
                    tensorized[key] = example[key]
            return tensorized

        examples = list(map(tensorize, examples))

        if not self.varlen:
            length_of_first = examples[0]["input_ids"].size(0)
            if all(example["input_ids"].size(0) == length_of_first for example in examples):
                batch = {
                    "input_ids": torch.stack([example["input_ids"] for example in examples], dim=0),
                }
            else:
                if self.tokenizer._pad_token is None:
                    raise ValueError(
                        f"You are attempting to pad samples but the tokenizer you are using "
                        f"({self.tokenizer.__class__.__name__}) does not have a pad token.",
                    )
                batch = self.tokenizer.pad(examples, return_tensors=self.return_tensors, return_attention_mask=False)
        else:
            if len(examples) > 1:
                raise ValueError("The batch size must be 1 for variable length inputs.")
            batch = {
                "input_ids": torch.cat([example["input_ids"] for example in examples], dim=0).unsqueeze(0),
            }
            if "offsets" in examples[0]:
                batch["offsets"] = torch.cat([example["offsets"] for example in examples], dim=0).unsqueeze(0)
            else:
                if self.tokenizer.add_bos_token:
                    offsets = []
                    if batch["input_ids"][0, 0] != self.tokenizer.bos_token_id:
                        offsets.append(torch.tensor([0], dtype=torch.long))
                    offsets.append(torch.where(batch["input_ids"].eq(self.tokenizer.bos_token_id))[1])
                    offsets.append(torch.tensor([len(batch["input_ids"][0])], dtype=torch.long))
                    batch["offsets"] = torch.cat(offsets, dim=0)
                elif self.tokenizer.add_eos_token:
                    offsets = [torch.tensor([0], dtype=torch.long)]
                    offsets.append(torch.where(batch["input_ids"].eq(self.tokenizer.eos_token_id))[1] + 1)
                    if batch["input_ids"][0, -1] != self.tokenizer.eos_token_id:
                        offsets.append(torch.tensor([len(batch["input_ids"][0])], dtype=torch.long))
                    batch["offsets"] = torch.cat(offsets, dim=0)
                else:
                    raise ValueError("You must allow the tokenizer to add either a bos or eos token as separators.")

        labels = batch["input_ids"].clone()
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch
