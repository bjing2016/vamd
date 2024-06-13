import torch.nn as nn
import torch
import numpy as np
from sit.layers import gelu, modulate, TimestepEmbedder, FinalLayer
from .mha import MultiheadAttention

class VAMDModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.linear_in = nn.Linear(3, args.embed_dim)
        self.t_embedder = TimestepEmbedder(args.embed_dim)
        if args.abs_pos_emb:
            self.register_buffer('pos_embed',
                nn.Parameter(torch.zeros(1, args.abs_pos_emb, args.embed_dim), requires_grad=False))
            
        self.layers = nn.ModuleList(
            [
                VAMDLayer(
                    embed_dim=args.embed_dim,
                    ffn_embed_dim=4 * args.embed_dim,
                    mha_heads=args.mha_heads,
                )
                for _ in range(args.num_layers)
            ]
        )
        self.linear_out = FinalLayer(args.embed_dim, 3)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize (and freeze) pos_embed by sin-cos embedding:
        if self.args.abs_pos_emb:
            pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.args.abs_pos_emb))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.layers:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.linear_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.linear_out.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.linear_out.linear.weight, 0)
        nn.init.constant_(self.linear_out.linear.bias, 0)


    def forward(self, x, t, mask):
        x = self.linear_in(x)
        t = self.t_embedder(t)
        if self.args.abs_pos_emb:
            x = x + self.pos_embed
        for layer in self.layers:
            x = layer(x, t, mask)
        return self.linear_out(x, t)

class VAMDLayer(nn.Module):
    """Transformer layer block."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        mha_heads,
        dropout=0.0,
        use_rotary_embeddings=False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.mha_heads = mha_heads
        self.inf = 1e5
        self.use_rotary_embeddings = use_rotary_embeddings
        self._init_submodules(add_bias_kv=True, dropout=dropout)

    def _init_submodules(self, add_bias_kv=False, dropout=0.0, ipa_args=None):

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.embed_dim, 6 * self.embed_dim, bias=True)
        )

        self.mha = AttentionWithRoPE(
            self.embed_dim,
            self.mha_heads,
            add_bias_kv=add_bias_kv,
            dropout=dropout,
            use_rotary_embeddings=self.use_rotary_embeddings,
        )

        self.mha_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

        self.fc1 = nn.Linear(self.embed_dim, self.ffn_embed_dim)
        self.fc2 = nn.Linear(self.ffn_embed_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, t, mask):
        shift_mha, scale_mha, gate_mha, \
            shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(t).chunk(6, dim=-1)

        residual = x
        x = modulate(self.mha_layer_norm(x), shift_mha, scale_mha)
        x = self.mha(x, mask)
        x = residual + gate_mha.unsqueeze(1) * x

        residual = x
        x = modulate(self.final_layer_norm(x), shift_mlp, scale_mlp)
        x = self.fc2(gelu(self.fc1(x)))
        x = residual + gate_mlp.unsqueeze(1) * x

        return x

class AttentionWithRoPE(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.attn = MultiheadAttention(*args, **kwargs)

    def forward(self, x, mask):
        x = x.transpose(0, 1)
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=1 - mask)
        x = x.transpose(0, 1)
        return x

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
