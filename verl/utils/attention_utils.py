# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable

import torch
import torch.nn.functional as F

_index_first_axis, _pad_input, _rearrange, _unpad_input = None, None, None, None


def _torch_index_first_axis(input_tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return input_tensor.index_select(0, indices.to(device=input_tensor.device, dtype=torch.long))


def _torch_pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    output_shape = (batch * seqlen, *hidden_states.shape[1:])
    output = hidden_states.new_zeros(output_shape)
    output.index_copy_(0, indices.to(device=hidden_states.device, dtype=torch.long), hidden_states)
    return output.reshape(batch, seqlen, *hidden_states.shape[1:])


def _torch_rearrange(input_tensor: torch.Tensor, pattern: str, **kwargs) -> torch.Tensor:
    if pattern == "b s ... -> (b s) ...":
        return input_tensor.reshape(input_tensor.shape[0] * input_tensor.shape[1], *input_tensor.shape[2:])
    if pattern == "(b s) ... -> b s ..." and "b" in kwargs and "s" in kwargs:
        return input_tensor.reshape(kwargs["b"], kwargs["s"], *input_tensor.shape[1:])
    try:
        from einops import rearrange as einops_rearrange

        return einops_rearrange(input_tensor, pattern, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"No fallback rearrange implementation for pattern: {pattern}") from exc


def _torch_unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor, unused_mask=None):
    del unused_mask
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).reshape(-1)
    max_seqlen_in_batch = seqlens_in_batch.max().item() if seqlens_in_batch.numel() > 0 else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(hidden_states.shape[0] * hidden_states.shape[1], *hidden_states.shape[2:])
    return _torch_index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen_in_batch


def _get_attention_functions() -> tuple[Callable, Callable, Callable, Callable]:
    """Dynamically import attention functions based on available hardware."""

    from verl.utils.device import is_torch_npu_available

    global _index_first_axis, _pad_input, _rearrange, _unpad_input

    if is_torch_npu_available(check_device=False):
        from verl.utils.npu_flash_attn_utils import index_first_axis, pad_input, rearrange, unpad_input
    else:
        try:
            from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
        except ModuleNotFoundError:
            index_first_axis, pad_input, rearrange, unpad_input = (
                _torch_index_first_axis,
                _torch_pad_input,
                _torch_rearrange,
                _torch_unpad_input,
            )

    _index_first_axis, _pad_input, _rearrange, _unpad_input = index_first_axis, pad_input, rearrange, unpad_input

    return _index_first_axis, _pad_input, _rearrange, _unpad_input


def index_first_axis(*args, **kwargs):
    """
    Unified entry point for `index_first_axis` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.index_first_axis`
      - On NPU: `transformers.integrations.npu_flash_attention.index_first_axis`
        (falls back to `transformers.modeling_flash_attention_utils._index_first_axis`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def pad_input(*args, **kwargs):
    """
    Unified entry point for `pad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.pad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.pad_input`
        (falls back to `transformers.modeling_flash_attention_utils._pad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    _, func, *_ = _get_attention_functions()
    return func(*args, **kwargs)


def rearrange(*args, **kwargs):
    """
    Unified entry point for `rearrange` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.rearrange`
      - On NPU: `transformers.integrations.npu_flash_attention.rearrange`
        (falls back to `einops.rearrange` if no dedicated NPU implementation exists).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func, _ = _get_attention_functions()
    return func(*args, **kwargs)


def unpad_input(*args, **kwargs):
    """
    Unified entry point for `unpad_input` across CUDA and NPU backends.

    Dynamically dispatches to the appropriate device-specific implementation:
      - On CUDA: `flash_attn.bert_padding.unpad_input`
      - On NPU: `transformers.integrations.npu_flash_attention.unpad_input`
        (falls back to `transformers.modeling_flash_attention_utils._unpad_input`
        in newer versions of transformers).

    Users can call this function directly without worrying about the underlying device.
    """
    *_, func = _get_attention_functions()
    return func(*args, **kwargs)


__all__ = ["index_first_axis", "pad_input", "rearrange", "unpad_input"]
