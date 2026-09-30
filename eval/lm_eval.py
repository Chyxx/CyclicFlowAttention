"""Evaluate CyclicFlowAttention with the current lm-eval harness."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import fla  # noqa: F401
import lm_eval
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

from eval.model_imports import import_model
from eval.task_queue import merge_results, parse_gpus, run_task_queue, table_result


RESULT_PREFIX = "__CYFA_LM_EVAL_RESULT__="


def _capture_cli_result(suppress_table: bool = False) -> dict | None:
    result = None
    simple_evaluate = lm_eval.simple_evaluate
    from lm_eval import utils

    make_table = utils.make_table

    def capture(*args, **kwargs):
        nonlocal result
        result = simple_evaluate(*args, **kwargs)
        return result

    lm_eval.simple_evaluate = capture
    if suppress_table:
        utils.make_table = lambda *args, **kwargs: ""
    try:
        cli_evaluate()
    finally:
        lm_eval.simple_evaluate = simple_evaluate
        utils.make_table = make_table
    return result


@register_model("cyfa")
class CyFALM(HFLM):
    def __init__(
        self,
        pretrained: str | None = None,
        custom_model_path: str | None = None,
        custom_model_module: str | None = None,
        **kwargs,
    ) -> None:
        import_model(pretrained, custom_model_path, custom_model_module)
        super().__init__(pretrained=pretrained, **kwargs)


def main() -> None:
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
    parser.add_argument("--max-length", type=int, default=0)
    parser.add_argument("--model-args-extra", default="")
    parser.add_argument("--limit", default="")
    parser.add_argument("--num-fewshot", default="")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args, extra = parser.parse_known_args()

    tasks = [task.strip() for task in args.tasks.split(",") if task.strip()]
    if not tasks:
        parser.error("--tasks must contain at least one task")
    if args.max_length < 0:
        parser.error("--max-length must be positive")
    if args.gpus and not args.worker:
        try:
            gpus = parse_gpus(args.gpus)
        except ValueError as error:
            parser.error(str(error))

        def command_for(task: str) -> list[str]:
            command = [
                sys.executable, "-m", "eval.lm_eval", "--worker",
                "--model-path", args.model_path,
                "--tasks", task,
                "--device", "cuda:0",
                "--batch-size", args.batch_size,
                "--dtype", args.dtype,
            ]
            if args.output_path:
                command += ["--output-path", str(Path(args.output_path) / task)]
            for option, value in (
                ("--custom-model-path", args.custom_model_path),
                ("--custom-model-module", args.custom_model_module),
                ("--tokenizer", args.tokenizer),
                ("--max-length", str(args.max_length) if args.max_length else ""),
                ("--model-args-extra", args.model_args_extra),
                ("--limit", args.limit),
                ("--num-fewshot", args.num_fewshot),
            ):
                if value:
                    command += [option, value]
            return command + extra

        try:
            results = run_task_queue(tasks, gpus, command_for, result_prefix=RESULT_PREFIX)
        except subprocess.CalledProcessError as error:
            raise SystemExit(error.returncode) from error
        from lm_eval.utils import make_table

        merged = merge_results([results[task] for task in tasks])
        print("\nCombined results:")
        print(make_table(merged))
        if "groups" in merged:
            print(make_table(merged, "groups"))
        return

    model_args = [
        f"pretrained={args.model_path}",
        f"dtype={args.dtype}",
        "trust_remote_code=true",
    ]
    for key, value in (
        ("tokenizer", args.tokenizer),
        ("custom_model_path", args.custom_model_path),
        ("custom_model_module", args.custom_model_module),
    ):
        if value:
            model_args.append(f"{key}={value}")
    if args.max_length:
        model_args.append(f"max_length={args.max_length}")
    if args.model_args_extra:
        model_args.extend(part.strip() for part in args.model_args_extra.split(",") if part.strip())

    sys.argv = [
        "lm_eval", "run",
        "--model", "cyfa",
        "--model_args", ",".join(model_args),
        "--tasks", ",".join(tasks),
        "--device", args.device,
        "--batch_size", args.batch_size,
        *extra,
    ]
    if args.output_path:
        sys.argv += ["--output_path", args.output_path]
    if args.limit:
        sys.argv += ["--limit", args.limit]
    if args.num_fewshot:
        sys.argv += ["--num_fewshot", args.num_fewshot]
    result = _capture_cli_result(suppress_table=args.worker)
    if args.worker and result is not None:
        from lm_eval.utils import handle_non_serializable

        print(
            RESULT_PREFIX + json.dumps(
                table_result(result),
                default=handle_non_serializable,
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
