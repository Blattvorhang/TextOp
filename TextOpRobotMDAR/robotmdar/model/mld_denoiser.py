import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# import clip
import loralib as lora


class DenoiserMLP(nn.Module):
    # =========================================================================
    # NOTE: DenoiserMLP is NOT currently used — the active config
    # (config/denoiser/def.yaml) uses DenoiserTransformer.  The MLP is kept as
    # a lighter alternative for ablations / memory-constrained runs.  It is
    # fully wired for goal + scene conditioning and will work out of the box
    # if you switch the config's _target_ to this class and add the matching
    # keys (goal_dim, grid_size, cond_goal_mask_prob, cond_scene_mask_prob).
    # =========================================================================

    def __init__(self,
                 h_dim=512,
                 n_blocks=2,
                 dropout: float = 0.1,
                 activation="gelu",
                 history_shape=(2, 276),
                 noise_shape=(1, 128),
                 goal_dim=5,
                 grid_size=25,
                 cond_goal_mask_prob=0.1,
                 cond_scene_mask_prob=0.1,
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.dropout = dropout
        self.n_blocks = n_blocks
        self.activation = activation

        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.goal_dim = goal_dim
        self.grid_size = grid_size
        self.scene_dim = grid_size**3
        self.cond_goal_mask_prob = cond_goal_mask_prob
        self.cond_scene_mask_prob = cond_scene_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(self.h_dim,
                                                       self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim,
                                               self.sequence_pos_encoder)

        self.embed_goal = nn.Linear(self.goal_dim, self.h_dim)
        self.embed_scene = nn.Linear(self.scene_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)

        # input: time + goal + scene + history + noise → all projected to h_dim
        input_dim = self.h_dim * 5
        self.input_project = nn.Linear(input_dim, self.h_dim)

        self.mlp = MLPBlock(h_dim=h_dim,
                            out_dim=np.prod(noise_shape),
                            n_blocks=n_blocks,
                            actfun=activation)

    def mask_condition(self, cond, probability, force_mask=False,
                       return_keep_mask=False):
        """Independent Bernoulli dropout for goal / scene conditions."""
        if force_mask:
            keep_mask = torch.zeros(cond.shape[0], dtype=torch.bool,
                                    device=cond.device)
        elif self.training and probability > 0.:
            drop_mask = torch.bernoulli(
                torch.full((cond.shape[0], 1), probability, device=cond.device)
            ).bool()
            keep_mask = ~drop_mask.squeeze(-1)
        else:
            keep_mask = torch.ones(cond.shape[0], dtype=torch.bool,
                                   device=cond.device)
        masked_cond = cond * keep_mask.unsqueeze(-1)
        if return_keep_mask:
            return masked_cond, keep_mask
        return masked_cond

    def forward(self, x_t, timesteps, y=None):
        """
        x_t: [B, T=1, D]
        timesteps: [batch_size] (int)
        y: dict with keys 'goal' [B, 5], 'voxel' [B, grid_size³],
           'history_motion_normalized' [B, T_hist, nfeats]
        """
        if y is None:
            raise ValueError(
                "Goal+scene denoiser requires a condition dictionary"
            )

        batch_size = x_t.shape[0]

        emb_time = self.embed_timestep(timesteps).squeeze(0)  # [bs, h_dim]

        goal, goal_keep_mask = self.mask_condition(
            y['goal'], self.cond_goal_mask_prob,
            force_mask=y.get('force_drop_goal', False),
            return_keep_mask=True)
        y['goal_condition_keep_mask'] = goal_keep_mask
        voxel = self.mask_condition(
            y['voxel'], self.cond_scene_mask_prob,
            force_mask=y.get('force_drop_scene', False))
        emb_goal = self.embed_goal(goal)     # [bs, h_dim]
        emb_scene = self.embed_scene(voxel)  # [bs, h_dim]

        emb_history = self.embed_history(
            y['history_motion_normalized'].reshape(
                batch_size, self.history_shape[-1]))  # [bs, h_dim]

        emb_noise = self.embed_noise(
            x_t.reshape(batch_size, self.noise_shape[-1]))  # [bs, h_dim]

        input_embed = torch.cat(
            (emb_time, emb_goal, emb_scene, emb_history, emb_noise), dim=1
        )  # [bs, input_dim]
        output = self.mlp(self.input_project(input_embed))  # [bs, noise_dim]
        output = output.reshape(batch_size, *self.noise_shape)

        return output


class DenoiserTransformer(nn.Module):

    def __init__(self,
                 h_dim=256,
                 ff_size=1024,
                 num_layers=4,
                 num_heads=4,
                 dropout=0.1,
                 activation="gelu",
                 history_shape=(2, 276),
                 noise_shape=(1, 128),
                 goal_dim=5,
                 grid_size=25,
                 cond_goal_mask_prob=0.1,
                 cond_scene_mask_prob=0.1,
                 use_vae=True,
                 **kargs):
        super().__init__()
        self.h_dim = h_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation

        self.history_shape = history_shape
        self.noise_shape = noise_shape
        self.goal_dim = goal_dim
        self.grid_size = grid_size
        self.scene_dim = grid_size**3
        self.cond_goal_mask_prob = cond_goal_mask_prob
        self.cond_scene_mask_prob = cond_scene_mask_prob

        # input embeddings
        self.sequence_pos_encoder = PositionalEncoding(self.h_dim,
                                                       self.dropout)
        self.embed_timestep = TimestepEmbedder(self.h_dim,
                                               self.sequence_pos_encoder)

        self.embed_goal = nn.Linear(self.goal_dim, self.h_dim)
        self.embed_scene = nn.Linear(self.scene_dim, self.h_dim)
        self.embed_history = nn.Linear(self.history_shape[-1], self.h_dim)
        self.embed_noise = nn.Linear(self.noise_shape[-1], self.h_dim)

        # transformer encoder layers
        print("TRANS_ENC init")
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=self.h_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation)
        self.seqTransEncoder = nn.TransformerEncoder(
            seqTransEncoderLayer, num_layers=self.num_layers)

        # output projection
        self.output_process = nn.Linear(self.h_dim, self.noise_shape[-1])

    def mask_condition(self, cond, probability, force_mask=False,
                       return_keep_mask=False):
        if force_mask:
            keep_mask = torch.zeros(cond.shape[0], dtype=torch.bool,
                                    device=cond.device)
        elif self.training and probability > 0.:
            drop_mask = torch.bernoulli(
                torch.full((cond.shape[0], 1), probability,
                           device=cond.device)
            ).bool()
            keep_mask = ~drop_mask.squeeze(-1)
        else:
            keep_mask = torch.ones(cond.shape[0], dtype=torch.bool,
                                   device=cond.device)
        masked_cond = cond * keep_mask.unsqueeze(-1)
        if return_keep_mask:
            return masked_cond, keep_mask
        return masked_cond

    def forward(self, x_t, timesteps, y=None):
        """
        x_t: [B, T=1, D]
        timesteps: [batch_size] (int)
        """
        if y is None:
            raise ValueError("Goal+scene denoiser requires a condition dictionary")

        emb_time = self.embed_timestep(timesteps)  # [1, bs, d]
        goal, goal_keep_mask = self.mask_condition(
            y['goal'], self.cond_goal_mask_prob,
            force_mask=y.get('force_drop_goal', False),
            return_keep_mask=True)
        y['goal_condition_keep_mask'] = goal_keep_mask
        voxel = self.mask_condition(
            y['voxel'], self.cond_scene_mask_prob,
            force_mask=y.get('force_drop_scene', False))
        emb_goal = self.embed_goal(goal).unsqueeze(0)
        emb_scene = self.embed_scene(voxel).unsqueeze(0)
        emb_history = self.embed_history(
            y['history_motion_normalized']).permute(1, 0,
                                                    2)  # [History, bs, d]
        emb_noise = self.embed_noise(x_t).permute(1, 0, 2)  # [1, bs, d]

        xseq = torch.cat(
            (emb_time, emb_goal, emb_scene, emb_history, emb_noise), dim=0
        )
        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqTransEncoder(xseq)[
            -self.noise_shape[0]:]  # [1, bs, h_dim]
        output = self.output_process(output)  # [1, B, noise_shape[-1]]
        output = output.permute(1, 0, 2)  # [B, 1, noise_shape[-1]]
        # print('output shape:', output.shape)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class TimestepEmbedder(nn.Module):

    def __init__(self, h_dim, sequence_pos_encoder):
        super().__init__()
        self.h_dim = h_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.h_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.h_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(
            self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class MLP(nn.Module):

    def __init__(self,
                 in_dim,
                 h_dims=[128, 128],
                 activation='tanh',
                 use_lora=False,
                 lora_rank=16):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        elif activation == 'gelu':
            self.activation = torch.nn.GELU()
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU()
        self.out_dim = h_dims[-1]
        self.layers = nn.ModuleList()
        in_dim_ = in_dim
        for h_dim in h_dims:
            layer = lora.Linear(in_dim_, h_dim,
                                r=lora_rank) if use_lora else nn.Linear(
                                    in_dim_, h_dim)
            self.layers.append(layer)
            in_dim_ = h_dim

    def forward(self, x):
        for fc in self.layers:
            x = self.activation(fc(x))
        return x


class MLPBlock(nn.Module):

    def __init__(self,
                 h_dim,
                 out_dim,
                 n_blocks,
                 actfun='relu',
                 residual=True,
                 use_lora=False,
                 lora_rank=16):
        super(MLPBlock, self).__init__()
        self.residual = residual
        self.layers = nn.ModuleList([
            MLP(h_dim, h_dims=(h_dim, h_dim), activation=actfun)
            for _ in range(n_blocks)
        ])  # two fc layers in each MLP
        self.out_fc = lora.Linear(h_dim, out_dim,
                                  r=lora_rank) if use_lora else nn.Linear(
                                      h_dim, out_dim)

    def forward(self, x):
        h = x
        for layer in self.layers:
            r = h if self.residual else 0
            h = layer(h) + r
        y = self.out_fc(h)
        return y
