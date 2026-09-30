"""Run independent evaluation tasks on a pool of GPUs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from queue import Empty, Queue


def parse_gpus(value: str) -> list[str]:
    gpus = [gpu.strip() for gpu in value.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one GPU id")
    if len(gpus) != len(set(gpus)):
        raise ValueError("--gpus contains duplicate GPU ids")
    return gpus


def run_task_queue(
    tasks: Sequence[str],
    gpus: Sequence[str | None],
    command_for: Callable[[str], list[str]],
    result_prefix: str = "",
    cwd: str | os.PathLike[str] | None = None,
) -> dict[str, dict]:
    pending: Queue[str] = Queue()
    results: dict[str, dict] = {}
    output_lock = threading.Lock()
    for task in tasks:
        pending.put(task)

    def run_gpu(gpu: str | None) -> None:
        env = os.environ.copy()
        if gpu is not None:
            env["CUDA_VISIBLE_DEVICES"] = gpu
        while True:
            try:
                task = pending.get_nowait()
            except Empty:
                return
            try:
                if not result_prefix:
                    subprocess.run(command_for(task), env=env, cwd=cwd, check=True)
                    continue

                process = subprocess.Popen(
                    command_for(task),
                    env=env,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                assert process.stdout is not None
                # tqdm redraws a progress line with '\r'. TextIOWrapper's default
                # universal-newline mode turns it into '\n', producing one terminal
                # line per update. Detect line boundaries without translating them.
                process.stdout.reconfigure(newline="")
                result = None
                for line in process.stdout:
                    marker = line.find(result_prefix)
                    if marker >= 0:
                        result = json.loads(line[marker + len(result_prefix):])
                        line = line[:marker]
                    if line:
                        with output_lock:
                            sys.stdout.write(line)
                            sys.stdout.flush()
                returncode = process.wait()
                if returncode:
                    raise subprocess.CalledProcessError(returncode, process.args)
                if result is None:
                    raise RuntimeError(f"task {task!r} produced no structured result")
                with output_lock:
                    results[task] = result
            finally:
                pending.task_done()

    with ThreadPoolExecutor(max_workers=min(len(gpus), len(tasks))) as executor:
        futures = [executor.submit(run_gpu, gpu) for gpu in gpus[: len(tasks)]]
        for future in futures:
            future.result()
    return results


def merge_results(results: Sequence[dict]) -> dict:
    """Merge lm-eval result dictionaries, including repeated task metrics."""
    merged: dict = {
        "results": {},
        "versions": {},
        "n-shot": {},
        "higher_is_better": {},
        "group_subtasks": {},
    }
    for result in results:
        for key in ("results", "groups"):
            for name, value in result.get(key, {}).items():
                if isinstance(value, dict):
                    merged.setdefault(key, {}).setdefault(name, {}).update(value)
                else:
                    merged.setdefault(key, {})[name] = value
        for name, value in result.get("higher_is_better", {}).items():
            merged["higher_is_better"].setdefault(name, {}).update(value)
        for key in ("versions", "n-shot", "group_subtasks", "configs"):
            if key in result:
                merged.setdefault(key, {}).update(result[key])
    return merged


def table_result(result: dict) -> dict:
    """Keep only fields consumed by lm-eval's terminal tables."""
    keys = (
        "results",
        "groups",
        "versions",
        "n-shot",
        "higher_is_better",
        "group_subtasks",
    )
    return {key: result[key] for key in keys if key in result}
