import torch
from torch import Tensor
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.nn.functional import _in_projection_packed, linear
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
import math
import numpy as np
from typing import Optional, Tuple
import warnings


class ProtDataset(data.Dataset):
    """
    Custom dataset for protein features and labels.

    This dataset handles input protein sequences and DMS scores for training and evaluating
    protein function prediction models. It supports accessing the length of the dataset
    and retrieving individual samples by index.

    Args:
        seqs (tensor): torch tensor of shape shape (N, L), contains extended tokenized 
                                protein sequences, where an AA \in [0, AA_size] at position k is 
                                coded as AA + AA_size x k, 
        labels (tensor): torch vector of length L, containing DMS score of individual variants.
        train (bool, optional): If True, indicates that the dataset is for training. 
                                Default is True.
    """
    
    def __init__(self, seqs, labels, train=True):    
        self.train = train
        self.seqs = seqs
        self.labels = labels
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        
        X = self.seqs[idx]
        y = self.labels[idx]
        
        return X, y
    
    
def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    """
    Computes the scaled dot-product attention for EpistaticTransformer

    This function is based on the original torch.nn.functional.scaled_dot_product_attention function.
    Key modifications have been made to allow the order of epistasis to scale as 2^M with M number of attention layers

    Args:
        query (Tensor): The query tensor of shape (L, N, E), where L is the target sequence length, 
                        N is the batch size, and E is the embedding dimension.
        key (Tensor): The key tensor of shape (S, N, E), where S is the source sequence length.
        value (Tensor): The value tensor of shape (S, N, E).
        attn_mask (Tensor, optional): An optional mask of shape (L, S) or (N * num_heads, L, S) to apply 
                                      before softmax. Default is None.
        dropout_p (float, optional): Dropout probability. Default is 0.0.

    Returns:
        Tensor: Output tensor of shape (L, N, E).
        Tensor: Attention weights of shape (N * num_heads, L, S).

    Note:
        The original function can be found at torch.nn.functional.scaled_dot_product_attention.
    """

    L, S = query.size(-2), key.size(-2)
    n = query.size(0)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype)
    device = query.device
    attn_bias = attn_bias.to(device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    num_heads = attn_weight.shape[1]
    attn_weight = attn_weight.transpose(0, 1)
    attn_weight = attn_weight.transpose(1, 2) # L first
    attn_weight = attn_weight.contiguous().view(num_heads, L, -1) # collapse to shape L x N L
    attn_weight = attn_weight.view(num_heads, L, n, L).transpose(1, 2).transpose(0, 1) # transform back to N x L x L
    attn_weight_ = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight_ @ value, attn_weight

def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> Tuple[Tensor, Optional[Tensor]]:
    """ Forward method for MultiHeadAttention, based on the original torch.nn.functional.scaled_dot_product_attention function.
    Key modifications have been made to allow the order of epistasis to scale as 2^M with M number of attention layers

    Note:
        The original function can be found in torch.nn.functional.py
    """

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)


    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #

    q = q.view(bsz, num_heads, tgt_len, head_dim)
    k = k.view(bsz, num_heads, src_len, head_dim)
    v = v.view(bsz, num_heads, src_len, head_dim)

    attn_output, attn_weights = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
    attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
    # if not is_batched:
    #     # squeeze the output if input was unbatched
    #     attn_output = attn_output.squeeze(1)
    return attn_output, attn_weights


class MultiheadAttention(nn.Module):
    """ Modified MultiheadAttention class, based on the original torch.nn.functional.scaled_dot_product_attention function.
    Key modifications have been made to allow the order of epistasis to scale as 2^M with M number of attention layers

    Note:
        The original function can be found in torch.nn.MultiheadAttention
    """
    
    
    r"""Allows the model to jointly attend to information
    from different representation subspaces as described in the paper:
    `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_.

    Multi-Head Attention is defined as:

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    ``nn.MultiHeadAttention`` will use the optimized implementations of
    ``scaled_dot_product_attention()`` when possible.

    In addition to support for the new ``scaled_dot_product_attention()``
    function, for speeding up Inference, MHA will use
    fastpath inference with support for Nested Tensors, iff:

    - self attention is being computed (i.e., ``query``, ``key``, and ``value`` are the same tensor).
    - inputs are batched (3D) with ``batch_first==True``
    - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor argument ``requires_grad``
    - training is disabled (using ``.eval()``)
    - ``add_bias_kv`` is ``False``
    - ``add_zero_attn`` is ``False``
    - ``batch_first`` is ``True`` and the input is batched
    - ``kdim`` and ``vdim`` are equal to ``embed_dim``
    - if a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ is passed, neither ``key_padding_mask``
      nor ``attn_mask`` is passed
    - autocast is disabled

    If the optimized inference fastpath implementation is in use, a
    `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be passed for
    ``query``/``key``/``value`` to represent padding more efficiently than using a
    padding mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_
    will be returned, and an additional speedup proportional to the fraction of the input
    that is padding can be expected.

    Args:
        embed_dim: Total dimension of the model.
        num_heads: Number of parallel attention heads. Note that ``embed_dim`` will be split
            across ``num_heads`` (i.e. each head will have dimension ``embed_dim // num_heads``).
        dropout: Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
        bias: If specified, adds bias to input / output projection layers. Default: ``True``.
        add_bias_kv: If specified, adds bias to the key and value sequences at dim=0. Default: ``False``.
        add_zero_attn: If specified, adds a new batch of zeros to the key and value sequences at dim=1.
            Default: ``False``.
        kdim: Total number of features for keys. Default: ``None`` (uses ``kdim=embed_dim``).
        vdim: Total number of features for values. Default: ``None`` (uses ``vdim=embed_dim``).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).

    Examples::

        >>> # xdoctest: +SKIP
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)

    .. _`FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness`:
         https://arxiv.org/abs/2205.14135

    """

    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
        is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
    Args:
        query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
            or :math:`(N, L, E_q)` when ``batch_first=True``, where :math:`L` is the target sequence length,
            :math:`N` is the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            Queries are compared against key-value pairs to produce the output.
            See "Attention Is All You Need" for more details.
        key: Key embeddings of shape :math:`(S, E_k)` for unbatched input, :math:`(S, N, E_k)` when ``batch_first=False``
            or :math:`(N, S, E_k)` when ``batch_first=True``, where :math:`S` is the source sequence length,
            :math:`N` is the batch size, and :math:`E_k` is the key embedding dimension ``kdim``.
            See "Attention Is All You Need" for more details.
        value: Value embeddings of shape :math:`(S, E_v)` for unbatched input, :math:`(S, N, E_v)` when
            ``batch_first=False`` or :math:`(N, S, E_v)` when ``batch_first=True``, where :math:`S` is the source
            sequence length, :math:`N` is the batch size, and :math:`E_v` is the value embedding dimension ``vdim``.
            See "Attention Is All You Need" for more details.
        key_padding_mask: If specified, a mask of shape :math:`(N, S)` indicating which elements within ``key``
            to ignore for the purpose of attention (i.e. treat as "padding"). For unbatched `query`, shape should be :math:`(S)`.
            Binary and float masks are supported.
            For a binary mask, a ``True`` value indicates that the corresponding ``key`` value will be ignored for
            the purpose of attention. For a float mask, it will be directly added to the corresponding ``key`` value.
        need_weights: If specified, returns ``attn_output_weights`` in addition to ``attn_outputs``.
            Set ``need_weights=False`` to use the optimized ``scaled_dot_product_attention``
            and achieve the best performance for MHA.
            Default: ``True``.
        attn_mask: If specified, a 2D or 3D mask preventing attention to certain positions. Must be of shape
            :math:`(L, S)` or :math:`(N\cdot\text{num\_heads}, L, S)`, where :math:`N` is the batch size,
            :math:`L` is the target sequence length, and :math:`S` is the source sequence length. A 2D mask will be
            broadcasted across the batch while a 3D mask allows for a different mask for each entry in the batch.
            Binary and float masks are supported. For a binary mask, a ``True`` value indicates that the
            corresponding position is not allowed to attend. For a float mask, the mask values will be added to
            the attention weight.
            If both attn_mask and key_padding_mask are supplied, their types should match.
        average_attn_weights: If true, indicates that the returned ``attn_weights`` should be averaged across
            heads. Otherwise, ``attn_weights`` are provided separately per head. Note that this flag only has an
            effect when ``need_weights=True``. Default: ``True`` (i.e. average weights across heads)
        is_causal: If specified, applies a causal mask as attention mask.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``attn_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Outputs:
        - **attn_output** - Attention outputs of shape :math:`(L, E)` when input is unbatched,
          :math:`(L, N, E)` when ``batch_first=False`` or :math:`(N, L, E)` when ``batch_first=True``,
          where :math:`L` is the target sequence length, :math:`N` is the batch size, and :math:`E` is the
          embedding dimension ``embed_dim``.
        - **attn_output_weights** - Only returned when ``need_weights=True``. If ``average_attn_weights=True``,
          returns attention weights averaged across heads of shape :math:`(L, S)` when input is unbatched or
          :math:`(N, L, S)`, where :math:`N` is the batch size, :math:`L` is the target sequence length, and
          :math:`S` is the source sequence length. If ``average_attn_weights=False``, returns attention weights per
          head of shape :math:`(\text{num\_heads}, L, S)` when input is unbatched or :math:`(N, \text{num\_heads}, L, S)`.

        .. note::
            `batch_first` argument is ignored for unbatched inputs.
        """

        why_not_fast_path = ''
        if ((attn_mask is not None and torch.is_floating_point(attn_mask))
           or (key_padding_mask is not None) and torch.is_floating_point(key_padding_mask)):
            why_not_fast_path = "floating-point masks are not supported for fast path."

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        if not is_batched:
            why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
        elif query is not key or key is not value:
            # When lifting this restriction, don't forget to either
            # enforce that the dtypes all match or test cases where
            # they don't!
            why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
        elif self.in_proj_bias is not None and query.dtype != self.in_proj_bias.dtype:
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_bias ({self.in_proj_bias.dtype}) don't match"
        elif self.in_proj_weight is None:
            why_not_fast_path = "in_proj_weight was None"
        elif query.dtype != self.in_proj_weight.dtype:
            # this case will fail anyway, but at least they'll get a useful error message.
            why_not_fast_path = f"dtypes of query ({query.dtype}) and self.in_proj_weight ({self.in_proj_weight.dtype}) don't match"
        elif self.training:
            why_not_fast_path = "training is enabled"
        elif (self.num_heads % 2) != 0:
            why_not_fast_path = "self.num_heads is not even"
        elif not self.batch_first:
            why_not_fast_path = "batch_first was not True"
        elif self.bias_k is not None:
            why_not_fast_path = "self.bias_k was not None"
        elif self.bias_v is not None:
            why_not_fast_path = "self.bias_v was not None"
        elif self.add_zero_attn:
            why_not_fast_path = "add_zero_attn was enabled"
        elif not self._qkv_same_embed_dim:
            why_not_fast_path = "_qkv_same_embed_dim was not True"
        elif query.is_nested and (key_padding_mask is not None or attn_mask is not None):
            why_not_fast_path = "supplying both src_key_padding_mask and src_mask at the same time \
                                 is not supported with NestedTensor input"
        elif torch.is_autocast_enabled():
            why_not_fast_path = "autocast is enabled"

        if not why_not_fast_path:
            tensor_args = (
                query,
                key,
                value,
                self.in_proj_weight,
                self.in_proj_bias,
                self.out_proj.weight,
                self.out_proj.bias,
            )
            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            if torch.overrides.has_torch_function(tensor_args):
                why_not_fast_path = "some Tensor argument has_torch_function"
            elif _is_make_fx_tracing():
                why_not_fast_path = "we are running make_fx tracing"
            elif not all(_check_arg_device(x) for x in tensor_args):
                why_not_fast_path = ("some Tensor argument's device is neither one of "
                                     f"cpu, cuda or {torch.utils.backend_registration._privateuse1_backend_name}")
            elif torch.is_grad_enabled() and any(_arg_requires_grad(x) for x in tensor_args):
                why_not_fast_path = ("grad is enabled and at least one of query or the "
                                     "input/output projection weights or biases requires_grad")
            if not why_not_fast_path:
                merged_mask, mask_type = self.merge_masks(attn_mask, key_padding_mask, query)

                if self.in_proj_bias is not None and self.in_proj_weight is not None:
                    return torch._native_multi_head_attention(
                        query,
                        key,
                        value,
                        self.embed_dim,
                        self.num_heads,
                        self.in_proj_weight,
                        self.in_proj_bias,
                        self.out_proj.weight,
                        self.out_proj.bias,
                        merged_mask,
                        need_weights,
                        average_attn_weights,
                        mask_type)

        any_nested = query.is_nested or key.is_nested or value.is_nested
        assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
                                f"The fast path was not hit because {why_not_fast_path}")

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights
    

    
def make_predictions(model, dataloader, pre_sig_forward=False):
    """
    Generate predictions using the provided model and dataloader.

    This function iterates over the given dataloader, passes the inputs through the model,
    and collects the predicted and true values. It can optionally use a model's 
    `pre_sig_forward` method if specified.

    Args:
        model (torch.nn.Module): The PyTorch model used for making predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader containing the input data and targets.
        pre_sig_forward (bool, optional): If True, use the model's `pre_sig_forward` method instead of 
                                          the standard forward method. Default is False.

    Returns:
        tuple: Two numpy arrays containing the predicted values (`pred_`) and the true values (`true_`).

    Example:
        >>> pred, true = make_predictions(model, dataloader, pre_sig_forward=True)

    Note:
        Ensure that the model and dataloader are compatible, and the model is in evaluation mode 
        (i.e., `model.eval()`) before calling this function to disable dropout and batch normalization.
    """
    
    pred, true = [], []
    with torch.no_grad():
        for batch_inputs, batch_targets in dataloader:
            if pre_sig_forward:
                output = model.pre_sig_forward(batch_inputs)
            else: 
                output = model(batch_inputs)
            target = batch_targets
            pred.append(output)
            true.append(target)
    pred_ = torch.concat(pred).flatten().cpu().detach().numpy()
    true_ = torch.concat(true).flatten().cpu().detach().numpy()
    return pred_, true_

def make_predictions_eval(model, dataloader):
    model.eval()
    pred, true, inputs = [], [], []
    for batch_inputs, batch_targets in dataloader:
        output = model(batch_inputs)
        target = batch_targets
        pred.append(output)
        true.append(target)
        inputs.append(batch_inputs)
    pred = torch.concat(pred).flatten().cpu().detach().numpy()
    true = torch.concat(true).flatten().cpu().detach().numpy()
    return pred, true, inputs


class LinearModel(nn.Module):
    """
    Model for fitting additive and nonspecific epistasis. 

    This model consists of a single fully connected layer, optional dropout, and
    an optional output normalization. The output is processed through a sigmoid 
    activation function and additional linear scaling.

    Args:
        L (int): The length of the input sequence.
        AA_size (int): The size of the input features.
        dropout_p (float): The dropout probability.
        fc_out_norm (bool): If True, applies batch normalization to the output of 
                            the fully connected layer.
    """    
    def __init__(self, L, AA_size, dropout_p, fc_out_norm):
        super(LinearModel, self).__init__()
        self.L = L
        self.AA_size = AA_size
        self.fc = nn.Linear((self.AA_size)*self.L, 1)
        self.dropout_p = dropout_p
        self.fc_out_norm = fc_out_norm
        self.sigmoid_norm = nn.BatchNorm1d(1, affine=False)                
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_scale = nn.Linear(1, 1)  
        self.fc_scale = nn.Linear(1, 1)
        
    def forward(self, x):
        n = len(x)
        x = x.view(n, -1)
        x = torch.dropout(x, self.dropout_p, train=True)        
        x = self.fc(x)
        if self.fc_out_norm == True:
            x = self.sigmoid_norm(x)
        x = self.fc_scale(x)
        x = self.sigmoid(x)
        x = self.sigmoid_scale(x)        
            
        return x
    
class Transformer_2k(nn.Module):
    """
    Transformer-based model for modeling higher order epistasis jointly with non-specific epistasis. 
    Highest order of specific epistasis fitted scales as 2^M, where M is the number of attetion heads

    Args:
        L (int): The length of the input sequence.
        input_dim (int): The size of the input vocabulary.
        hidden_dim (int): The dimensionality of the hidden layers.
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        dropout_p (float): The dropout probability.

    Attributes:
        dropout (nn.Dropout): Dropout layer.
        embedding (nn.Embedding): Embedding layer for input sequences.
        transformer_layers (nn.ModuleList): List of transformer layers with multi-head attention.
        hidden_dim (int): The dimensionality of the hidden layers.
        input_dim (int): The size of the input vocabulary.
        num_layers (int): The number of transformer layers.
        num_heads (int): The number of attention heads in each transformer layer.
        dropout_p (float): The dropout probability.
        fc (nn.Linear): Fully connected layer for the final output.
        fc_add (nn.Linear): Additional fully connected layer for the final output.
        phi_norm (nn.BatchNorm1d): Batch normalization layer for the final output.
        phi_scale (nn.Linear): Linear layer to scale the batch-normalized output.
        sigmoid (nn.Sigmoid): Sigmoid activation function.
        sigmoid_scale (nn.Linear): Linear layer to scale the sigmoid output.

    Methods:
        forward(x):
            Defines the forward pass of the model. Takes input tensor `x`, applies
            embedding, transformer layers, dropout, fully connected layers, batch
            normalization, and sigmoid activation with additional linear scaling.
        pre_sig_forward(x):
            Defines a forward pass without the sigmoid activation and scaling. Useful 
            for pre-sigmoid outputs.
    """    
    def __init__(self, L, input_dim, hidden_dim, num_layers, num_heads, dropout_p):
        super(Transformer_2k, self).__init__()
        
        self.dropout = torch.nn.Dropout(dropout_p)
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer_layers = nn.ModuleList([
            MultiheadAttention(hidden_dim, num_heads, dropout=dropout_p)
            for _ in range(num_layers)
        ])
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        self.fc = nn.Linear(hidden_dim*L, 1)
        self.fc_add = nn.Linear(hidden_dim*L, 1)        
        self.phi_norm = nn.BatchNorm1d(1, affine=False)  
        self.phi_scale = nn.Linear(1, 1)        
        self.sigmoid = nn.Sigmoid()
        self.sigmoid_scale = nn.Linear(1, 1)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, L).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1) after applying embedding,
                    transformer layers, dropout, fully connected layers, batch normalization,
                    and sigmoid activation with additional linear scaling.
        """        
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # seq_len x batch x hidden_dim
        v0 = x.clone()
        
        for layer in self.transformer_layers:
            x = layer(x, x, v0)[0] + x
            
        x = x.permute(1, 0, 2)
        x = x.flatten(1)
        x = self.dropout(x)
        
        v0 = v0.permute(1, 0, 2)
        v0 = v0.flatten(1)
        v0 = self.dropout(v0)
        
        x = self.fc(x) + self.fc_add(v0) # batch x 1 (scalar)
        x = self.phi_norm(x)
        x = self.phi_scale(x)
        x = self.sigmoid(x)
        x = self.sigmoid_scale(x)
        return x
            
    def pre_sig_forward(self, x):

        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # seq_len x batch x hidden_dim
        v0 = x.clone()
        for layer in self.transformer_layers:
            x = layer(x, x, v0)[0] + x
        x = x.permute(1, 0, 2)
        x = x.flatten(1)
        x = self.dropout(x)
        v0 = v0.permute(1, 0, 2)
        v0 = v0.flatten(1)
        v0 = self.dropout(v0)
        x = self.fc(x) + self.fc_add(v0) # batch x 1 (scalar)

        return x
