# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# Derived from Flash Linear Attention and licensed under the MIT license
# found in the LICENSE file in this directory.
# For a list of upstream contributors, visit:
# https://github.com/fla-org/flash-linear-attention/graphs/contributors

from __future__ import annotations

import math

import torch
import torch.distributed as dist
from accelerate.utils import DistributedType
from datasets import Dataset, DatasetDict, load_from_disk
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import Partial, Replicate
from transformers import AutoTokenizer, Trainer

from cyclic_flow_attention import CyclicFlowAttentionConfig, CyclicFlowAttentionForCausalLM

from .data import DataCollatorForLanguageModeling
from .logging import LogCallback, get_logger
from .parser import get_train_args


logger = get_logger(__name__)


def clip_fsdp2_grad_norm_(parameters, max_norm: float) -> torch.Tensor:
    """Clip the true global norm of FSDP2-sharded gradients."""
    local_grads = []
    local_norms = []
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        if isinstance(grad, DTensor):
            if any(isinstance(placement, Partial) for placement in grad.placements):
                raise RuntimeError("FSDP2 produced a partial gradient before gradient clipping.")
            local_grad = grad.to_local()
            replication = math.prod(
                grad.device_mesh.shape[mesh_dim]
                for mesh_dim, placement in enumerate(grad.placements)
                if isinstance(placement, Replicate)
            )
        else:
            local_grad = grad
            replication = dist.get_world_size()
        local_grads.append(local_grad)
        local_norms.append(torch.linalg.vector_norm(local_grad.detach().float()) / math.sqrt(replication))

    if not local_norms:
        return torch.tensor(0.0, device=torch.cuda.current_device())

    total_norm_sq = torch.stack(local_norms).square().sum()
    dist.all_reduce(total_norm_sq, op=dist.ReduceOp.SUM)
    total_norm = total_norm_sq.sqrt()
    clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
    for grad in local_grads:
        grad.mul_(clip_coef)
    return total_norm


class CyclicFlowAttentionTrainer(Trainer):
    def _clip_grad_norm(self, model):
        if self.accelerator.distributed_type == DistributedType.FSDP and self.accelerator.is_fsdp2:
            self.accelerator.unscale_gradients()
            return clip_fsdp2_grad_norm_(model.parameters(), self.args.max_grad_norm)
        return super()._clip_grad_norm(model)


def main() -> None:
    args = get_train_args()
    logger.info(args)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        use_fast=args.use_fast_tokenizer,
        trust_remote_code=True,
        add_bos_token=True,
        add_eos_token=False,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.from_config:
        config = CyclicFlowAttentionConfig.from_pretrained(args.model_name_or_path)
        model = CyclicFlowAttentionForCausalLM(config)
    else:
        model = CyclicFlowAttentionForCausalLM.from_pretrained(args.model_name_or_path)
    model.accepts_loss_kwargs = False
    model.train()

    if args.fsdp:
        args.fsdp_plugin_args["transformer_cls_names_to_wrap"] = model._no_split_modules
        args.fsdp_plugin_args["mixed_precision_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )

    trainable = model.num_parameters(only_trainable=True)
    total = model.num_parameters()
    logger.info("Trainable parameters: %d / %d (%.2f%%)", trainable, total, 100 * trainable / total)
    logger.info("Loading tokenized dataset from %s", args.cache_dir)
    dataset = load_from_disk(args.cache_dir)
    if isinstance(dataset, DatasetDict):
        if args.split not in dataset:
            raise ValueError(f"Split {args.split!r} is missing; available splits: {list(dataset)}")
        dataset = dataset[args.split]
    if not isinstance(dataset, Dataset) or "input_ids" not in dataset.column_names:
        raise ValueError("Training requires a tokenized Dataset with an input_ids column.")
    dataset = dataset.shuffle(seed=args.seed)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, varlen=args.varlen)

    if args.lr_scheduler_type == "cosine_with_min_lr":
        if not any(key in args.lr_scheduler_kwargs for key in ("min_lr", "min_lr_rate")):
            args.lr_scheduler_kwargs["min_lr_rate"] = 0.1

    trainer = CyclicFlowAttentionTrainer(
        model=model,
        args=args,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        train_dataset=dataset,
    )
    trainer.model_accepts_loss_kwargs = False
    results = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    if trainer.accelerator.distributed_type == DistributedType.FSDP:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(trainer.args.output_dir)
    trainer.log_metrics("train", results.metrics)
    trainer.save_metrics("train", results.metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
