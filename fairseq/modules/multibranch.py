import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from . import MultiheadAttention

class MultiBranch(nn.Module):
    def __init__(self, branches, embed_dim_list):
        super().__init__()
        self.branches = nn.ModuleList(branches)
        self.embed_dim_list = embed_dim_list

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None):
        tgt_len, bsz, embed_size = query.size()
        assert sum(self.embed_dim_list) == embed_size
        out = []
        attn = None
        start = 0
        for idx, embed_dim in enumerate(self.embed_dim_list):
            branch = self.branches[idx]
            branch_type = type(branch)

            q = query[...,start:start+embed_dim]
            if key is not None:
                assert value is not None
                k, v = key[..., start:start+embed_dim], value[..., start:start+embed_dim]
            start += embed_dim

            if branch_type == MultiheadAttention:
                 x, attn = branch(q, k, v, key_padding_mask, incremental_state, need_weights, static_kv, attn_mask)
            else:
                mask = key_padding_mask
                if mask is not None:
                    q = q.masked_fill(mask.transpose(0, 1).unsqueeze(2), 0)
                x = branch(q.contiguous(), incremental_state=incremental_state)
            out.append(x)

        out = torch.cat(out, dim=-1)
        return out, attn