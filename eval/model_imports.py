"""Load FLA-compatible model code at the evaluation boundary."""

from __future__ import annotations

import importlib
import json
import os
import sys
from functools import wraps
from types import ModuleType

import transformers
from transformers.modeling_utils import PreTrainedModel


if int(transformers.__version__.split(".", 1)[0]) < 5:
    raise RuntimeError("CyclicFlowAttention evaluation requires Transformers 5 or newer")


def _model_type(model_name_or_path: str | None) -> str | None:
    if not model_name_or_path:
        return None
    path = model_name_or_path
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle).get("model_type")


def import_fla_model(model_name_or_path: str | None) -> None:
    import fla  # noqa: F401

    model_type = _model_type(model_name_or_path)
    if model_type:
        module_name = f"fla.models.{model_type}"
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # Custom architectures are registered by their own package and do
            # not necessarily have a matching module inside FLA.
            if error.name != module_name:
                raise
        else:
            _patch(module)


def import_model(
    model_name_or_path: str | None,
    custom_model_path: str | None = None,
    custom_model_module: str | None = None,
) -> None:
    """Register CyFA and optional custom code, then patch the loaded model."""
    _patch(importlib.import_module("cyclic_flow_attention"))
    if custom_model_path or custom_model_module:
        import_custom_model(custom_model_path, custom_model_module)
    import_fla_model(model_name_or_path)


def import_custom_model(path: str | None, module_name: str | None = None) -> None:
    if not path:
        if module_name:
            _patch(importlib.import_module(module_name))
        return

    path = os.path.abspath(path)
    if os.path.isfile(path):
        search_path = os.path.dirname(path)
        module_name = module_name or os.path.splitext(os.path.basename(path))[0]
    elif os.path.isfile(os.path.join(path, "__init__.py")):
        search_path = os.path.dirname(path)
        module_name = module_name or os.path.basename(path)
    elif os.path.isdir(path) and module_name:
        search_path = path
    else:
        raise ValueError(f"Cannot import a model package from {path}")

    if search_path not in sys.path:
        sys.path.insert(0, search_path)
    _patch(importlib.import_module(module_name))


def _patch(module: ModuleType) -> None:
    for value in vars(module).values():
        if isinstance(value, ModuleType) and value.__name__.startswith(f"{module.__name__}."):
            _patch(value)
            continue
        if not isinstance(value, type) or value is PreTrainedModel:
            continue
        if isinstance(getattr(value, "_tied_weights_keys", None), list):
            value._tied_weights_keys = {}
        if issubclass(value, PreTrainedModel):
            _patch_generation(value)


def _patch_generation(model_class: type[PreTrainedModel]) -> None:
    prepare = getattr(model_class, "prepare_inputs_for_generation", None)
    if prepare is None or getattr(prepare, "_cyfa_recurrent_cache", False):
        return

    @wraps(prepare)
    def prepare_inputs_for_generation(self, *args, **kwargs):
        cache = kwargs.get("past_key_values")
        get_seq_length = getattr(cache, "get_seq_length", None)
        if get_seq_length is not None and int(get_seq_length()) == 0:
            kwargs["past_key_values"] = None
        return prepare(self, *args, **kwargs)

    prepare_inputs_for_generation._cyfa_recurrent_cache = True
    model_class.prepare_inputs_for_generation = prepare_inputs_for_generation
