# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the real VL weight loader without constructing an NPU model."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_visual_projection_matches_rotated_embedding_basis(dtype, rotated):
    source_path = ROOT / "vllm_ascend/models/minimax_m3/minimax_m3_vl.py"
    model_class = next(
        node
        for node in ast.parse(source_path.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MiniMaxM3SparseForConditionalGeneration"
    )
    method = next(node for node in model_class.body if getattr(node, "name", None) == "load_weights")
    # A nonsymmetric orthogonal matrix detects using Q rather than Q.T on W.
    rotation = torch.tensor([[0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]], dtype=torch.float32)
    recorded = {}

    class RecordingLoader:
        def __init__(self, model):
            assert model is target

        def load_weights(self, weights, *, mapper):
            recorded["weights"] = dict(weights)
            recorded["mapper"] = mapper
            return set(recorded["weights"])

    rotation_path = Path("quarot.safetensors") if rotated else None
    target = SimpleNamespace(_vision_rotation_path=rotation_path, hf_to_vllm_mapper=object())
    load_rotation = Mock(return_value=rotation)
    namespace = {
        "AutoWeightsLoader": RecordingLoader,
        "get_rotation_matrix": load_rotation,
        "logger": Mock(),
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source_path), "exec"), namespace)

    weight_key = "patch_merge_mlp.linear_2.weight"
    bias_key = "patch_merge_mlp.linear_2.bias"
    incoming = {
        weight_key: (torch.arange(12).reshape(4, 3) / 7).to(dtype),
        bias_key: torch.tensor([0.1, -0.2, 0.3, 0.4], dtype=dtype),
        "patch_merge_mlp.linear_1.weight": torch.eye(3, dtype=dtype),
        "language_model.model.embed_tokens.weight": torch.eye(4, dtype=dtype),
    }
    original = {key: value.clone() for key, value in incoming.items()}
    loaded = namespace["load_weights"](target, iter(incoming.items()))
    actual = recorded["weights"]
    assert loaded == set(incoming)
    assert recorded["mapper"] is target.hf_to_vllm_mapper

    inputs = torch.tensor([[0.5, -0.25, 1.0], [-1.0, 0.5, 0.25]])
    expected_features = inputs @ original[weight_key].float().T + original[bias_key].float()
    if rotated:
        expected_features = expected_features @ rotation
        load_rotation.assert_called_once_with(rotation_path)
    else:
        load_rotation.assert_not_called()
    actual_features = inputs @ actual[weight_key].float().T + actual[bias_key].float()
    torch.testing.assert_close(actual_features, expected_features)

    for key in incoming:
        assert actual[key].dtype == dtype
        assert torch.equal(incoming[key], original[key]), key
        if not rotated or key not in {weight_key, bias_key}:
            assert actual[key] is incoming[key], key
