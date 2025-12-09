# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False  # Llama 3.2 uses scaled RoPE

    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    # MoT (Mixture-of-Transformers) parameters
    use_mot: bool = False
    n_modalities: int = 2  # Number of modalities (e.g., text, image, speech)
    qk_normalization: bool = False  # Whether to use QK normalization
    debug_mot: bool = False  # Enable debug output for MoT routing
    
    # Audio Expert parameters
    use_audio_expert: bool = False  # Whether to use a specialized audio expert (e.g., MusicGen)
    audio_expert_dim: int = 1024    # Dimension of the audio expert (MusicGen Small=1024)


class MoTAdapter(nn.Module):
    """
    Adapter to project input to a smaller dimension for a specialized expert,
    then project back to the model dimension.
    """
    def __init__(self, input_dim: int, expert_dim: int, expert_module: nn.Module):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, expert_dim, bias=False)
        self.expert = expert_module
        self.up_proj = nn.Linear(expert_dim, input_dim, bias=False)

    def forward(self, x):
        # Down-project: [..., input_dim] -> [..., expert_dim]
        x_small = self.down_proj(x)
        # Expert processing: [..., expert_dim] -> [..., expert_dim]
        out_small = self.expert(x_small)
        # Up-project: [..., expert_dim] -> [..., input_dim]
        return self.up_proj(out_small)


class MusicGenFeedForward(nn.Module):
    """
    Standard FeedForward layer used in MusicGen (GELU activation, 2 linear layers).
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # Note: using standard Linear instead of ColumnParallel/RowParallel 
        # for simplicity in this specialized expert, but could be parallelized.
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


# MoT utility functions
def merge_modalities(
    expert_outputs: List[torch.Tensor], modality_masks: List[torch.Tensor]
) -> torch.Tensor:
    """
    Merge modality-specific outputs into a unified tensor.
    
    Args:
        expert_outputs: List of modality-specific outputs. Each has shape [num_tokens, D] 
                       where num_tokens is the number of tokens for that modality.
        modality_masks: List of boolean masks. Each mask has shape [bs, seq_len] indicating 
                       which positions belong to that modality.
    
    Returns:
        Merged tensor of shape [bs, seq_len, D] where D matches expert_outputs[0].shape[-1].
    """
    assert len(expert_outputs) == len(modality_masks)
    assert len(expert_outputs) > 0
    
    if len(expert_outputs) == 1:
        # Single modality: need to reshape from [num_tokens, D] to [bs, seq_len, D]
        mask = modality_masks[0]
        bs, seq_len = mask.shape
        dim = expert_outputs[0].shape[-1]
        merged = torch.zeros(bs, seq_len, dim, dtype=expert_outputs[0].dtype, 
                            device=expert_outputs[0].device)
        merged[mask] = expert_outputs[0]
        return merged
    
    # Get output shape from first expert
    dim = expert_outputs[0].shape[-1]
    mask = modality_masks[0]
    bs, seq_len = mask.shape
    
    # Initialize merged tensor
    merged = torch.zeros(bs, seq_len, dim, dtype=expert_outputs[0].dtype,
                        device=expert_outputs[0].device)
    
    # Merge in reverse order to handle overlapping masks correctly
    for i in range(len(expert_outputs) - 1, -1, -1):
        expert_output = expert_outputs[i]  # [num_tokens, dim]
        mask = modality_masks[i]  # [bs, seq_len]
        merged[mask] = expert_output
    
    return merged


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        # KV cache (will be moved to correct device in forward pass)
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        # Detach to prevent graph connections across iterations
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(
            1, 2
        )  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class MoTAttention(nn.Module):
    """
    Mixture-of-Transformers Attention layer with modality-specific projections.
    Uses global attention computation but modality-specific Q, K, V, O projections.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.n_modalities = args.n_modalities
        self._debug = getattr(args, 'debug_mot', False)  # Enable debug output
        
        # Create modality-specific query, key, value, and output projections
        self.local_experts_wq = nn.ModuleList([
            ColumnParallelLinear(
                args.dim,
                args.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            for _ in range(self.n_modalities)
        ])
        self.local_experts_wk = nn.ModuleList([
            ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            for _ in range(self.n_modalities)
        ])
        self.local_experts_wv = nn.ModuleList([
            ColumnParallelLinear(
                args.dim,
                self.n_kv_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x,
            )
            for _ in range(self.n_modalities)
        ])
        self.local_experts_wo = nn.ModuleList([
            RowParallelLinear(
                args.n_heads * self.head_dim,
                args.dim,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x,
            )
            for _ in range(self.n_modalities)
        ])
        
        # QK normalization (if enabled)
        if args.qk_normalization:
            self.local_experts_q_normalization = nn.ModuleList([
                RMSNorm(self.head_dim, eps=args.norm_eps)
                for _ in range(self.n_modalities)
            ])
            self.local_experts_k_normalization = nn.ModuleList([
                RMSNorm(self.head_dim, eps=args.norm_eps)
                for _ in range(self.n_modalities)
            ])
        else:
            self.local_experts_q_normalization = None
            self.local_experts_k_normalization = None
        
        # Final output normalization for each modality
        self.local_experts_attention_norm = nn.ModuleList([
            RMSNorm(args.dim, eps=args.norm_eps) for _ in range(self.n_modalities)
        ])
        
        # KV cache (shared across modalities)
        # Will be moved to correct device in forward pass
        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        modality_masks: List[torch.Tensor],
    ):
        """
        Args:
            x: Input tensor of shape [bs, seq_len, dim]
            start_pos: Starting position for KV cache
            freqs_cis: Rotary embedding frequencies
            mask: Attention mask
            modality_masks: List of boolean masks, each of shape [bs, seq_len]
        """
        bsz, seqlen, _ = x.shape
        
        # Process Q, K, V for each modality
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = self._process_qkv(
            x, modality_masks
        )
        
        # Merge modality-specific Q, K, V into unified tensors
        xq = merge_modalities(expert_outputs_xq, modality_masks)  # [bs, seqlen, n_heads * head_dim]
        xk = merge_modalities(expert_outputs_xk, modality_masks)
        xv = merge_modalities(expert_outputs_xv, modality_masks)
        
        # Reshape for attention computation
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        
        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        
        # Update KV cache
        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)
        # Detach to prevent graph connections across iterations
        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk.detach()
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv.detach()

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]
        
        # Repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)
        
        # Compute attention
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        attn_output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        
        # Process final output with modality-specific projections and normalization
        output = self._process_final_output(attn_output, modality_masks)
        
        return output
    
    def _process_qkv(self, x: torch.Tensor, modality_masks: List[torch.Tensor]):
        """Process query, key, and value projections for each modality."""
        expert_outputs_xq, expert_outputs_xk, expert_outputs_xv = [], [], []
        
        for i in range(self.n_modalities):
            mask = modality_masks[i]  # [bs, seq_len]
            num_tokens = mask.sum().item()
            
            # Debug: show which expert is being used (only for first layer to reduce spam)
            if hasattr(self, '_debug') and self._debug and hasattr(self, '_layer_id') and self._layer_id == 0:
                print(f"  [MoT Debug Layer 0] Expert {i}: processing {num_tokens}/{mask.shape[1]} tokens")
            
            expert_input = x[mask]  # [num_tokens, dim]
            
            # Project through modality-specific layers
            xq = self.local_experts_wq[i](expert_input)
            xk = self.local_experts_wk[i](expert_input)
            xv = self.local_experts_wv[i](expert_input)
            
            # Apply QK normalization if enabled
            if self.local_experts_q_normalization is not None:
                # Reshape for normalization: [num_tokens, n_heads * head_dim] -> [num_tokens, n_heads, head_dim]
                num_tokens = xq.shape[0]
                n_heads_total = xq.shape[-1] // self.head_dim
                xq_reshaped = xq.view(num_tokens, n_heads_total, self.head_dim)
                xk_reshaped = xk.view(num_tokens, self.n_local_kv_heads, self.head_dim)
                
                # Normalize each head
                xq_normalized = torch.stack([
                    self.local_experts_q_normalization[i](xq_reshaped[:, h, :])
                    for h in range(n_heads_total)
                ], dim=1).view(num_tokens, -1)
                xk_normalized = torch.stack([
                    self.local_experts_k_normalization[i](xk_reshaped[:, h, :])
                    for h in range(self.n_local_kv_heads)
                ], dim=1).view(num_tokens, -1)
                
                xq = xq_normalized
                xk = xk_normalized
            
            expert_outputs_xq.append(xq)
            expert_outputs_xk.append(xk)
            expert_outputs_xv.append(xv)
        
        return expert_outputs_xq, expert_outputs_xk, expert_outputs_xv
    
    def _process_final_output(self, output: torch.Tensor, modality_masks: List[torch.Tensor]):
        """Process final attention output with modality-specific wo projections and normalization."""
        expert_outputs = []
        
        for i in range(self.n_modalities):
            mask = modality_masks[i]  # [bs, seq_len]
            expert_input = output[mask]  # [num_tokens, n_heads * head_dim]
            
            # Project through modality-specific output layer
            expert_output = self.local_experts_wo[i](expert_input)
            
            # Note: Post-normalization removed - we use pre-normalization in MoTTransformerBlock instead
            # This matches the standard Llama architecture better
            # expert_output = self.local_experts_attention_norm[i](expert_output)
            
            expert_outputs.append(expert_output)
        
        # Merge outputs back into original sequence order
        merged_output = merge_modalities(expert_outputs, modality_masks)
        return merged_output


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )
        self.w2 = RowParallelLinear(
            hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x
        )
        self.w3 = ColumnParallelLinear(
            dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x
        )

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoTFeedForward(nn.Module):
    """
    Mixture-of-Transformers FeedForward layer with modality-specific experts.
    """
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        n_modalities: int = 2,
        debug: bool = False,
        use_audio_expert: bool = False,
        audio_expert_dim: int = 1024,
    ):
        super().__init__()
        self.n_modalities = n_modalities
        self._debug = debug
        self.use_audio_expert = use_audio_expert
        
        experts = []
        for i in range(self.n_modalities):
            # Check if this should be an audio expert (MusicGen)
            # We arbitrarily assign the last expert to be the audio expert if enabled
            is_audio_expert = use_audio_expert and (i == self.n_modalities - 1)
            
            if is_audio_expert:
                # Use MusicGen specific FFN architecture
                # MusicGen Small: 1024 -> 4096 -> 1024
                # For simplicity we assume hidden_dim is 4*dim unless specified
                audio_ffn = MusicGenFeedForward(
                    dim=audio_expert_dim,
                    hidden_dim=4 * audio_expert_dim, 
                )
                
                # Wrap in adapter
                expert = MoTAdapter(
                    input_dim=dim,
                    expert_dim=audio_expert_dim,
                    expert_module=audio_ffn
                )
            else:
                # Standard Llama Expert
                expert = FeedForward(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    multiple_of=multiple_of,
                    ffn_dim_multiplier=ffn_dim_multiplier,
                )
            
            experts.append(expert)

        # Create modality-specific feed-forward experts
        self.local_experts = nn.ModuleList(experts)
        
        # Modality-specific normalization layers
        self.local_experts_ffn_norm = nn.ModuleList([
            RMSNorm(dim, eps=1e-5) for _ in range(self.n_modalities)
        ])

    def forward(
        self,
        x: torch.Tensor,
        modality_masks: List[torch.Tensor],
    ):
        """
        Args:
            x: Input tensor of shape [bs, seq_len, dim]
            modality_masks: List of boolean masks, each of shape [bs, seq_len]
        """
        expert_outputs = []
        for i in range(self.n_modalities):
            # Extract tokens for this modality
            mask = modality_masks[i]  # [bs, seq_len]
            num_tokens = mask.sum().item()
            
            # Debug: show which expert is being used (only for first layer)
            if self._debug and hasattr(self, '_layer_id') and self._layer_id == 0:
                print(f"  [MoT Debug Layer 0] FFN Expert {i}: processing {num_tokens}/{mask.shape[1]} tokens")
            
            expert_input = x[mask]  # [num_tokens, dim]
            
            # Process through modality-specific FFN
            expert_output = self.local_experts[i](expert_input)
            
            # Note: Post-normalization removed - we use pre-normalization in MoTTransformerBlock instead
            # expert_output = self.local_experts_ffn_norm[i](expert_output)
            
            expert_outputs.append(expert_output)
        
        # Merge outputs back into original sequence order
        merged_output = merge_modalities(expert_outputs, modality_masks)
        return merged_output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class MoTTransformerBlock(nn.Module):
    """
    Mixture-of-Transformers Transformer block with modality-specific components.
    Note: Normalization is handled within MoTAttention and MoTFeedForward modules
    (modality-specific normalization after projections).
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.layer_id = layer_id
        
        # Use MoT components
        self.attention = MoTAttention(args)
        self.attention._layer_id = layer_id  # For debug output
        self.feed_forward = MoTFeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            n_modalities=args.n_modalities,
            debug=getattr(args, 'debug_mot', False),
            use_audio_expert=getattr(args, 'use_audio_expert', False),
            audio_expert_dim=getattr(args, 'audio_expert_dim', 1024),
        )
        self.feed_forward._layer_id = layer_id  # For debug output
        
        # Add pre-normalization layers (like standard TransformerBlock)
        # These normalize BEFORE routing to experts
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        modality_masks: List[torch.Tensor],
    ):
        """
        Args:
            x: Input tensor of shape [bs, seq_len, dim]
            start_pos: Starting position for KV cache
            freqs_cis: Rotary embedding frequencies
            mask: Attention mask
            modality_masks: List of boolean masks, each of shape [bs, seq_len]
        """
        # Pre-normalize before routing to experts (like standard model)
        x_norm = self.attention_norm(x)
        
        # Debug: Check for NaN after attention norm
        if torch.isnan(x_norm).any():
            print(f"   ⚠️  NaN detected after attention_norm in layer {self.layer_id}")
        
        # MoT attention (modality-specific normalization also happens inside MoTAttention after projections)
        attn_out = self.attention(x_norm, start_pos, freqs_cis, mask, modality_masks)
        
        # Debug: Check for NaN after attention
        if torch.isnan(attn_out).any():
            print(f"   ⚠️  NaN detected after attention in layer {self.layer_id}")
        
        h = x + attn_out
        
        # Pre-normalize before FFN
        h_norm = self.ffn_norm(h)
        
        # Debug: Check for NaN after FFN norm
        if torch.isnan(h_norm).any():
            print(f"   ⚠️  NaN detected after ffn_norm in layer {self.layer_id}")
        
        # MoT feed-forward (modality-specific normalization also happens inside MoTFeedForward after FFN)
        ffn_out = self.feed_forward(h_norm, modality_masks)
        
        # Debug: Check for NaN after FFN
        if torch.isnan(ffn_out).any():
            print(f"   ⚠️  NaN detected after feed_forward in layer {self.layer_id}")
        
        out = h + ffn_out
        
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(
            params.vocab_size, params.dim, init_method=lambda x: x
        )

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            if params.use_mot:
                self.layers.append(MoTTransformerBlock(layer_id, params))
            else:
                self.layers.append(TransformerBlock(layer_id, params))
        
        # Note: MoT experts will be initialized from checkpoint weights after load_state_dict
        # See _initialize_mot_experts_from_checkpoint() method

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(
            params.dim, params.vocab_size, bias=False, init_method=lambda x: x
        )

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
        )
    
    def _initialize_mot_experts_from_checkpoint(self, checkpoint: dict):
        """
        Initialize MoT modality experts by copying weights from standard model layers.
        This ensures MoT experts start with pretrained weights rather than random initialization.
        """
        copied_count = 0
        for layer_id, layer in enumerate(self.layers):
            if isinstance(layer, MoTTransformerBlock):
                # Get standard layer prefix
                std_prefix = f"layers.{layer_id}"
                
                # Copy attention weights to all modality experts
                for mod_idx in range(self.params.n_modalities):
                    # Q, K, V, O projections
                    for proj_name in ["wq", "wk", "wv", "wo"]:
                        std_key = f"{std_prefix}.attention.{proj_name}.weight"
                        
                        if std_key in checkpoint:
                            # Copy standard weights to modality expert
                            with torch.no_grad():
                                expert_layer = getattr(layer.attention, f"local_experts_{proj_name}")[mod_idx]
                                # Handle FairScale parallel layers - access the underlying weight
                                if hasattr(expert_layer, 'weight'):
                                    expert_layer.weight.copy_(checkpoint[std_key])
                                    copied_count += 1
                
                # Copy FFN weights to all modality experts
                for mod_idx in range(self.params.n_modalities):
                    for ffn_name in ["w1", "w2", "w3"]:
                        std_key = f"{std_prefix}.feed_forward.{ffn_name}.weight"

                        if std_key in checkpoint:
                            with torch.no_grad():
                                expert = layer.feed_forward.local_experts[mod_idx]

                                # Check if this is a MoTAdapter (wrapped audio expert)
                                if isinstance(expert, MoTAdapter):
                                    # For MusicGenFeedForward, map w1->fc1, w3->fc2
                                    if ffn_name == "w1" and hasattr(expert.expert, 'fc1'):
                                        # Skip copying w1 to audio expert (different architecture)
                                        continue
                                    elif ffn_name == "w3" and hasattr(expert.expert, 'fc2'):
                                        # Skip copying w3 to audio expert (different architecture)
                                        continue
                                    elif ffn_name == "w2":
                                        # MusicGen doesn't have w2 (gate projection)
                                        continue
                                else:
                                    # Standard expert with w1, w2, w3
                                    expert_layer = getattr(expert, ffn_name)
                                    if hasattr(expert_layer, 'weight'):
                                        expert_layer.weight.copy_(checkpoint[std_key])
                                        copied_count += 1
                
                # Copy normalization layer weights
                # Pre-attention norm
                std_attn_norm_key = f"{std_prefix}.attention_norm.weight"
                if std_attn_norm_key in checkpoint:
                    with torch.no_grad():
                        layer.attention_norm.weight.copy_(checkpoint[std_attn_norm_key])
                        copied_count += 1
                
                # Pre-FFN norm
                std_ffn_norm_key = f"{std_prefix}.ffn_norm.weight"
                if std_ffn_norm_key in checkpoint:
                    with torch.no_grad():
                        layer.ffn_norm.weight.copy_(checkpoint[std_ffn_norm_key])
                        copied_count += 1
        
        print(f"   ✅ Copied {copied_count} weight tensors to MoT experts")

    # @torch.inference_mode()  # Commented out for training - gradients needed
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        modality_masks: Optional[List[torch.Tensor]] = None,
    ):
        """
        Args:
            tokens: Input token ids of shape [bs, seq_len]
            start_pos: Starting position for KV cache
            modality_masks: Optional list of boolean masks for MoT, each of shape [bs, seq_len].
                           If None and use_mot=False, all tokens are treated as a single modality.
                           If None and use_mot=True, defaults to all tokens being modality 0.
        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        
        # Debug: Check for NaN in embeddings
        if torch.isnan(h).any():
            print(f"   ⚠️  NaN detected in embeddings! Token range: [{tokens.min().item()}, {tokens.max().item()}]")
            print(f"      Embedding stats: min={h.min().item():.4f}, max={h.max().item():.4f}, mean={h.mean().item():.4f}")
            print(f"      NaN count: {torch.isnan(h).sum().item()}")
        
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack(
                [torch.zeros((seqlen, start_pos), device=tokens.device), mask]
            ).type_as(h)

        # Handle modality masks for MoT
        if self.params.use_mot:
            if modality_masks is None:
                # Default: all tokens belong to modality 0
                modality_masks = [
                    torch.ones(_bsz, seqlen, dtype=torch.bool, device=tokens.device)
                ] + [
                    torch.zeros(_bsz, seqlen, dtype=torch.bool, device=tokens.device)
                    for _ in range(self.params.n_modalities - 1)
                ]
            
            # Ensure we have the right number of masks
            assert len(modality_masks) == self.params.n_modalities, \
                f"Expected {self.params.n_modalities} modality masks, got {len(modality_masks)}"
            
            for layer in self.layers:
                if isinstance(layer, MoTTransformerBlock):
                    h = layer(h, start_pos, freqs_cis, mask, modality_masks)
                else:
                    # Fallback for non-MoT layers
                    h = layer(h, start_pos, freqs_cis, mask)
        else:
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
        
        h = self.norm(h)
        output = self.output(h).float()
        return output
