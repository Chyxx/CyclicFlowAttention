"""Run prefix-recall tasks with the current Transformers runtime."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "eval" / "vendor" / "lm_eval_prefix"
RESULT_PREFIX = "__CYFA_RECALL_RESULT__="


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", "--pretrained", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--custom-model-path", default="")
    parser.add_argument("--custom-model-module", default="")
    parser.add_argument("--tokenizer", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpus", default="", help="Comma-separated GPUs; run one task per GPU worker.")
    parser.add_argument("--batch-size", default="32")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--answer-length", type=int, default=48)
    parser.add_argument("--limit", default="")
    parser.add_argument("--write-out", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _worker(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(VENDOR))
    import torch
    from eval.model_imports import import_model
    from lm_eval.api.registry import register_model
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    import_model(args.model_path, args.custom_model_path, args.custom_model_module)

    @register_model("cyfa_recall")
    class RecallLM(HFLM):
        def __init__(self, checkpoint_name, dtype="bfloat16", device="cuda", tokenizer=None, **kwargs):
            torch_dtype = None if dtype == "auto" else getattr(torch, dtype)
            model = AutoModelForCausalLM.from_pretrained(checkpoint_name, dtype=torch_dtype)
            model.to(device=device) if torch_dtype is None else model.to(device=device, dtype=torch_dtype)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer or checkpoint_name)
            tokenizer.pad_token = tokenizer.eos_token
            super().__init__(
                pretrained=model,
                tokenizer=tokenizer,
                backend="causal",
                max_length=args.context_length,
                device=device,
                **kwargs,
            )

    model_args = [f"checkpoint_name={args.model_path}", f"dtype={args.dtype}"]
    if args.tokenizer:
        model_args.append(f"tokenizer={args.tokenizer}")
    sys.argv = [
        "lm_eval",
        "--model", "cyfa_recall",
        "--model_args", ",".join(model_args),
        "--tasks", args.tasks,
        "--device", args.device,
        "--batch_size", args.batch_size,
        "--num_fewshot", "0",
        "--context_length", str(args.context_length),
        "--answer_length", str(args.answer_length),
        "--context_key", "text",
        "--cutting_context",
    ]
    if args.output_path:
        sys.argv += ["--output_path", args.output_path, "--log_samples"]
    if args.limit:
        sys.argv += ["--limit", args.limit]
    if args.write_out:
        sys.argv.append("--write_out")
    from lm_eval import evaluator
    from lm_eval import __main__ as recall_main

    result = None
    simple_evaluate = evaluator.simple_evaluate
    make_table = recall_main.make_table

    def capture(*capture_args, **capture_kwargs):
        nonlocal result
        result = simple_evaluate(*capture_args, **capture_kwargs)
        return result

    evaluator.simple_evaluate = capture
    recall_main.make_table = lambda *table_args, **table_kwargs: ""
    try:
        recall_main.cli_evaluate()
    finally:
        evaluator.simple_evaluate = simple_evaluate
        recall_main.make_table = make_table
    if result is not None:
        from eval.task_queue import table_result

        print(
            RESULT_PREFIX + json.dumps(
                table_result(result),
                default=recall_main._handle_non_serializable,
                ensure_ascii=False,
            ),
            flush=True,
        )


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    if args.worker:
        _worker(args)
        return

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        parser.error("--tasks must contain at least one task")
    for task in tasks:
        if not (task.startswith("based_") or task == "scrolls"):
            parser.error(f"{task!r} is not a recall task; use eval.lm_eval")

    def command_for(task: str) -> list[str]:
        command = [
            sys.executable, "-m", "eval.recall_eval", "--worker",
            "--model-path", args.model_path,
            "--tasks", task,
            "--device", "cuda:0" if args.gpus else args.device,
            "--batch-size", args.batch_size,
            "--dtype", args.dtype,
            "--context-length", str(args.context_length),
            "--answer-length", str(args.answer_length),
        ]
        if args.output_path:
            command += ["--output-path", str(Path(args.output_path) / task)]
        for option, value in (
            ("--custom-model-path", args.custom_model_path),
            ("--custom-model-module", args.custom_model_module),
            ("--tokenizer", args.tokenizer),
            ("--limit", args.limit),
        ):
            if value:
                command += [option, value]
        if args.write_out:
            command.append("--write-out")
        return command

    from eval.task_queue import merge_results, parse_gpus, run_task_queue
    if args.gpus:
        try:
            gpus = parse_gpus(args.gpus)
        except ValueError as error:
            parser.error(str(error))
    else:
        gpus = [None]
    try:
        results = run_task_queue(
            tasks,
            gpus,
            command_for,
            result_prefix=RESULT_PREFIX,
            cwd=ROOT,
        )
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode) from error

    # Import the current lm-eval only in the parent, after vendor workers exit.
    from lm_eval.utils import make_table

    merged = merge_results([results[task] for task in tasks])
    print("\nCombined results:")
    print(make_table(merged))
    if "groups" in merged:
        print(make_table(merged, "groups"))


if __name__ == "__main__":
    main()
