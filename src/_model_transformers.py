import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList
from torch.nn.modules.activation import MultiheadAttention

from st_moe_pytorch import MoE
from st_moe_pytorch import SparseMoEBlock

import math, copy
from typing import Optional, Union, Callable

from tqdm import tqdm

from transformers import T5EncoderModel

# ------------------------------------------------------------------------- #

__all__ = ['Transformer', 'TransformerDecoder', 'TransformerDecoderLayer']

# ------------------------------------------------------------------------- #

def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_seq_len(src: torch.Tensor, batch_first: bool) -> Optional[int]:
    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            return src_size[0]
        else:
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]

def _get_activation_fn(activation: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError(f'activation should be relu/gelu, not {activation}')

def _generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32

    return torch.triu(torch.full((sz, sz), float('-inf'), dtype = dtype, device = device), diagonal = 1)

def _detect_is_causal_mask(mask: Optional[torch.Tensor], is_causal: Optional[bool] = None, size: Optional[int] = None) -> bool:
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(sz, device = mask.device, dtype = mask.dtype)

        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal

# ------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout = 0.1, max_len = 5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_parameter('pe', nn.Parameter(pe, requires_grad = False))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, dtype: torch.dtype = torch.bfloat16):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device = freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim = -1)

    return cache.to(dtype = dtype)

@torch.jit.script
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x = x.permute(0, 2, 1, 3)
    d = x.shape[-1] // 2

    cos = freqs_cis[..., 0][None, :, None]
    sin = freqs_cis[..., 1][None, :, None]

    x1, x2 = x[..., :d], x[..., d : d * 2]
    tmp = x1.clone()

    x1_new = x1.mul(cos) - x2.mul(sin)
    x2_new = x2.mul(cos) + tmp.mul(sin)

    x = torch.cat((x1_new, x2_new), dim = -1)
    x = x.permute(0, 2, 1, 3)
    
    return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 batch_first: bool = True,
                 device: Optional[torch.device] = None,
                 dtype: Optional[torch.dtype] = None):
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()

        self.embed_dim = embed_dim
        self.batch_first = batch_first
        self.dim_head = embed_dim // num_heads
        self.scale = self.dim_head ** -0.5
        self.heads = num_heads
        hidden_dim = self.dim_head * num_heads

        self.to_qkv = nn.Linear(embed_dim, hidden_dim * 3, bias = False, **factory_kwargs)
        self.to_out = nn.Linear(hidden_dim, embed_dim, bias = False, **factory_kwargs)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = True) -> torch.Tensor:
        if not self.batch_first:
            x = x.transpose(0, 1)

        b, n, _ = x.size()
        q, k, v = torch.chunk(self.to_qkv(x), chunks = 3, dim = -1)
        q, k, v = map(lambda t: t.contiguous().view(b, self.heads, n, -1), (q, k, v))

        self.freqs_cis = precompute_freqs_cis(seq_len = n, n_elem = self.embed_dim // self.heads, base = 10000, dtype = x.dtype).to(x.device)
        freqs_cis = self.freqs_cis[: x.shape[1]]

        # q = apply_rotary_emb(q, freqs_cis)
        # k = apply_rotary_emb(k, freqs_cis)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal = is_causal)
        out = out.contiguous().view(b, n, -1)
        out = self.dropout(out)

        return self.to_out(out)
    
# ------------------------------------------------------------------------- #

class Transformer(nn.Module):
    def __init__(self,
                 n_vocab: int = 30000,
                 d_model: int = 512,
                 nhead: int = 8,
                 max_len: int = 5000,
                 num_decoder_layers: int = 6,
                 dim_feedforward: int = 2048,
                 use_moe: bool = False, 
                 num_experts: int = 16,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = True,
                 norm_first: bool = False,
                 bias: bool = True,
                 device = None,
                 dtype = None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()

        torch._C._log_api_usage_once(f'torch.nn.modules.{self.__class__.__name__}')

        self.use_moe = use_moe

        self.input_emb = nn.Embedding(n_vocab, d_model, **factory_kwargs)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len).to(device)

        # Load the KE-T5 encoder
        self.encoder = T5EncoderModel.from_pretrained('KETI-AIR/ke-t5-base-ko').to(device)
        # Freeze the encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        decoder_layer = TransformerDecoderLayer(d_model,
                                                nhead,
                                                dim_feedforward,
                                                use_moe,
                                                num_experts,
                                                dropout,
                                                activation,
                                                layer_norm_eps,
                                                batch_first,
                                                norm_first,
                                                bias,
                                                **factory_kwargs)
        decoder_norm = nn.LayerNorm(d_model, eps = layer_norm_eps, bias = bias, **factory_kwargs)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, use_moe, decoder_norm)

        self.projection = nn.Linear(d_model, n_vocab).to(device)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.batch_first = batch_first

    def forward(self,
                src: torch.Tensor,
                src_mask: torch.Tensor,
                tgt: torch.Tensor,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_is_causal: bool = True,
                memory_is_causal: bool = False) -> torch.Tensor:
        
        if src.dim() != tgt.dim():
            raise RuntimeError('the number of dimensions in src and tgt must be equal')

        memory = self.encoder(src, attention_mask = src_mask).last_hidden_state

        tgt = self.input_emb(tgt) * math.sqrt(self.d_model)
        tgt = self.pos_encoder(tgt)
        
        if self.use_moe:
            with torch.amp.autocast(device_type = 'cuda', enabled = False):
                output, sum_total_aux_loss = self.decoder(tgt,
                                                          memory,
                                                          memory_mask = memory_mask,
                                                          memory_key_padding_mask=memory_key_padding_mask,
                                                          tgt_is_causal = tgt_is_causal,
                                                          memory_is_causal = memory_is_causal)
        else:
            output = self.decoder(tgt,
                                  memory,
                                  memory_mask = memory_mask,
                                  memory_key_padding_mask = memory_key_padding_mask,
                                  tgt_is_causal = tgt_is_causal,
                                  memory_is_causal = memory_is_causal)
        
        output = self.projection(output)

        if self.use_moe:
            return output, sum_total_aux_loss
        else:
            return output
        
    def generate(self, src: torch.Tensor, src_mask: torch.Tensor, max_len: int = 100, temperature: float = 1.0):
        if src.dim() != 2:
            raise RuntimeError('The src tensor should be 2-dimensional')
        
        tgt_fin = torch.full((src.size(0), 1), 1, dtype = torch.long, device=src.device)

        for i in tqdm(range(max_len)):
            max_index = tgt_fin.max()

            tgt = tgt_fin
            if self.use_moe:
                output, _ = self.froward(src,
                                         src_mask,
                                         tgt,
                                         memory_mask = None,
                                         memory_key_padding_mask = None,
                                         tgt_is_causal = True,
                                         memory_is_causal = False)
            else:
                output = self.forward(src,
                                      src_mask,
                                      tgt,
                                      memory_mask = None,                                
                                      memory_key_padding_mask = None,
                                      tgt_is_causal = True,
                                      memory_is_causal = False)
                
            logits = output
            output = F.log_softmax(logits / temperature, dim = -1)
            output = output.view(-1, output.size(-1))

            next_tokens = torch.multinomial(torch.exp(output), 1)[-1]
            tgt_fin = torch.cat((tgt_fin, next_tokens.unsqueeze(-1)), dim = 1)

        return tgt_fin[:, 1:]

    @staticmethod
    def generate_square_subsequent_mask(sz: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return _generate_square_subsequent_mask(sz, dtype = dtype, device = device)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

class TransformerDecoder(nn.Module):
    __constants__ = ['norm']

    def __init__(self,
                 decoder_layer: 'TransformerDecoderLayer',
                 num_layers: int,
                 use_moe: bool = False,
                 norm: Optional[nn.Module] = None) -> None:
        
        super().__init__()

        torch._C._log_api_usage_once(f'torch.nn.modules.{self.__class__.__name__}')

        self.layers = _get_clones(decoder_layer, num_layers)

        self.num_layers = num_layers
        self.use_moe = use_moe
        self.norm = norm

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False) -> torch.Tensor:
        
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        if self.use_moe:
            sum_total_aux_loss = 0
            for mod in self.layers:
                output, total_aux_loss, balance_loss, router_z_loss = mod(output,
                                                                          memory,
                                                                          memory_mask = memory_mask,
                                                                          memory_key_padding_mask = memory_key_padding_mask,
                                                                          tgt_is_causal = tgt_is_causal,
                                                                          memory_is_causal = memory_is_causal)
                
                sum_total_aux_loss += total_aux_loss
        else:
            for mod in self.layers:
                output = mod(output,
                             memory,
                             memory_mask = memory_mask,
                             memory_key_padding_mask = memory_key_padding_mask,
                             tgt_is_causal = tgt_is_causal,
                             memory_is_causal = memory_is_causal)

        if self.norm is not None:
            output = self.norm(output)

        if self.use_moe:
            return output, sum_total_aux_loss
        else:
            return output

class TransformerDecoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int = 2048,
                 use_moe: bool = False,
                 num_experts: int = 16,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[torch.Tensor], torch.Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 bias: bool = True,
                 device = None,
                 dtype = None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}

        super().__init__()

        self.self_attn = MultiHeadSelfAttention(d_model, nhead, dropout = dropout, batch_first = batch_first, **factory_kwargs) 
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout = dropout, batch_first = batch_first, bias = bias, **factory_kwargs)
        self.use_moe = use_moe

        if use_moe:
            self.moe = MoE(dim = d_model,
                           num_experts = num_experts,
                           gating_top_n = 2,
                           threshold_train = 0.2,
                           threshold_eval = 0.2,
                           capacity_factor_train = 1.25,
                           capacity_factor_eval = 2.,
                           balance_loss_coef = 1e-2,
                           router_z_loss_coef = 1e-3).to(device)
            
            self.moe_block = SparseMoEBlock(self.moe, add_ff_before = True, add_ff_after = True).to(device)

        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias = bias, **factory_kwargs)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model, bias = bias, **factory_kwargs)

        self.norm_first = norm_first

        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps, bias = bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps, bias = bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps = layer_norm_eps, bias = bias, **factory_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)


    def forward(self,
                tgt: torch.Tensor,
                memory: torch.Tensor,
                memory_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False) -> torch.Tensor:
        
        x = tgt
        
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)

            if self.use_moe:
                m, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)
                x = x + m
            else:
                x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))

            if self.use_moe:
                m, total_aux_loss, balance_loss, router_z_loss = self.moe_block(x)
                x = x + m
            else:
                x = self.norm3(x + self._ff_block(x))

        if self.use_moe:
            return x, total_aux_loss, balance_loss, router_z_loss
        else:
            return x

    def _sa_block(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        x = self.self_attn(x, is_causal = is_causal)
        return self.dropout1(x)

    def _mha_block(self,
                   x: torch.Tensor,
                   mem: torch.Tensor,
                   attn_mask: Optional[torch.Tensor],
                   key_padding_mask: Optional[torch.Tensor],
                   is_causal: bool = False) -> torch.Tensor:
        
        x = self.multihead_attn(x, mem, mem,
                                attn_mask = attn_mask,
                                key_padding_mask = key_padding_mask,
                                is_causal = is_causal,
                                need_weights = False)[0]
        
        return self.dropout2(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)