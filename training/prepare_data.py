# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Derived from Flash Linear Attention and licensed under the MIT license
# found in the LICENSE file in this directory.
# For a list of upstream contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import argparse
import os
from itertools import chain
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize and pack a text dataset for CyFA pretraining.")
    parser.add_argument("--dataset", required=True, help="Hugging Face dataset name or save_to_disk directory.")
    parser.add_argument("--dataset-config", default=None, help="Optional Hugging Face dataset configuration.")
    parser.add_argument("--data-files", default=None, help="Optional local file or glob passed to load_dataset.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="fla-hub/gla-1.3B-100B")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--download-cache", default=None)
    parser.add_argument("--num-proc", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument("--map-batch-size", type=int, default=1000)
    return parser.parse_args()


def load_source(args: argparse.Namespace) -> Dataset:
    path = Path(args.dataset)
    if path.is_dir() and (path / "state.json").is_file():
        source = load_from_disk(path)
    elif path.is_dir() and (path / "dataset_dict.json").is_file():
        source = load_from_disk(path)
    else:
        kwargs = {"split": args.split, "cache_dir": args.download_cache}
        if args.data_files:
            kwargs["data_files"] = args.data_files
        source = load_dataset(args.dataset, args.dataset_config, **kwargs)
    if isinstance(source, DatasetDict):
        if args.split not in source:
            raise ValueError(f"Split {args.split!r} is not present in {args.dataset!r}.")
        source = source[args.split]
    if not isinstance(source, Dataset):
        raise TypeError("Preprocessing requires a non-streaming Hugging Face Dataset.")
    if args.text_column not in source.column_names:
        raise ValueError(
            f"Text column {args.text_column!r} is missing; available columns: {source.column_names}",
        )
    return source


def main() -> None:
    args = parse_args()
    if args.context_length <= 0:
        raise ValueError("--context-length must be positive")
    if args.num_proc <= 0:
        raise ValueError("--num-proc must be positive")
    if args.map_batch_size <= 0:
        raise ValueError("--map-batch-size must be positive")

    source = load_source(args)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=True,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False,
    )

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return {
            "input_ids": tokenizer(
                batch[args.text_column],
                add_special_tokens=True,
                return_attention_mask=False,
                return_token_type_ids=False,
            )["input_ids"],
        }

    tokenized = source.map(
        tokenize,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_proc,
        remove_columns=source.column_names,
        desc="Tokenizing",
    )

    def pack(batch: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        tokens = list(chain.from_iterable(batch["input_ids"]))
        usable = len(tokens) // args.context_length * args.context_length
        return {
            "input_ids": [
                tokens[offset : offset + args.context_length]
                for offset in range(0, usable, args.context_length)
            ],
        }

    packed = tokenized.map(
        pack,
        batched=True,
        batch_size=args.map_batch_size,
        num_proc=args.num_proc,
        desc=f"Packing {args.context_length}-token samples",
    )
    if len(packed) == 0:
        raise ValueError("The source dataset did not produce a complete training sample.")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Output path already exists: {output_dir}")
    packed.save_to_disk(output_dir, num_proc=args.num_proc)
    print(f"Saved {len(packed):,} samples ({len(packed) * args.context_length:,} tokens) to {output_dir}")


if __name__ == "__main__":
    main()
