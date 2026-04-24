"""
Deep-learning architectures for Shadow Limb ankle trajectory prediction.

Three model variants are provided:

1. **AnkleLSTM** (Encoder-only): Bi-directional LSTM encodes a 200ms IMU window,
   then a linear head predicts the ankle angle at the next timestep.
   Simpler, lower latency, good baseline.

2. **AnkleEncoderDecoder** (Full Encoder-Decoder): Bi-directional encoder
   produces a context vector; a unidirectional decoder autoregressively
   generates a short future trajectory (~50ms).

3. **AnkleTransformer** (Self-attention baseline): Projects the IMU window
   into a latent space, adds learned positional embeddings, and runs a
   Transformer encoder. The representation of the most recent timestep is
   fed to a linear head to predict the ankle angle. This is the direct
   counterpart to AnkleLSTM for the LSTM-vs-Transformer comparison.
"""

import math

import torch
import torch.nn as nn
from . import config


class AnkleLSTM(nn.Module):
    """
    Bi-directional LSTM encoder with a linear regression head.

    Input:  (batch, seq_len, num_channels)
    Output:
      - (batch,)    when output_dim == 1 (direct angle regression)
      - (batch, 2)  when output_dim == 2 (phase (sin, cos) regression)

    When ``normalize_output=True`` the final output is L2-normalized along
    the feature dimension so that (sin, cos) predictions stay on the unit
    circle — the parameterization the Reference-Gait-Tracking pipeline
    expects before recovering φ via atan2.
    """

    def __init__(
        self,
        input_dim: int = config.NUM_INPUT_CHANNELS,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LSTM_LAYERS,
        dropout: float = config.DROPOUT,
        bidirectional: bool = config.BIDIRECTIONAL,
        output_dim: int = 1,
        normalize_output: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_dim = output_dim
        self.normalize_output = normalize_output

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.dropout = nn.Dropout(dropout)

        encoder_out_dim = hidden_dim * self.num_directions
        self.head = nn.Sequential(
            nn.Linear(encoder_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)

        # h_n shape: (num_layers * num_directions, batch, hidden_dim)
        # Take the last layer's hidden states from both directions
        if self.bidirectional:
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            context = torch.cat([h_forward, h_backward], dim=-1)
        else:
            context = h_n[-1]

        context = self.dropout(context)
        out = self.head(context)

        if self.normalize_output:
            # L2-normalize (sin, cos) to stay on the unit circle.
            # Clamp by a small epsilon to avoid division by zero at init.
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return out  # keep (batch, output_dim)

        if self.output_dim == 1:
            return out.squeeze(-1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AnkleEncoderDecoder(nn.Module):
    """
    Full Encoder-Decoder LSTM for multi-step trajectory prediction.

    Encoder: Bi-directional LSTM compresses a 200ms IMU window into context Z.
    Decoder: Unidirectional LSTM takes Z and autoregressively generates
             ankle angles for the next `horizon` timesteps.

    Input:  (batch, seq_len, num_channels)
    Output: (batch, horizon)  -- predicted ankle trajectory
    """

    def __init__(
        self,
        input_dim: int = config.NUM_INPUT_CHANNELS,
        hidden_dim: int = config.HIDDEN_DIM,
        num_layers: int = config.NUM_LSTM_LAYERS,
        dropout: float = config.DROPOUT,
        horizon: int = config.HORIZON_SAMPLES,
        bidirectional_encoder: bool = config.BIDIRECTIONAL,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.horizon = horizon
        self.enc_directions = 2 if bidirectional_encoder else 1

        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional_encoder,
        )

        # Project encoder's bidirectional hidden/cell to decoder's unidirectional shape
        enc_h_dim = hidden_dim * self.enc_directions
        self.h_projection = nn.Linear(enc_h_dim, hidden_dim)
        self.c_projection = nn.Linear(enc_h_dim, hidden_dim)

        # Decoder LSTM: input is the previous predicted angle (dim=1)
        self.decoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def _bridge_hidden(
        self, h_enc: torch.Tensor, c_enc: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transform encoder hidden states into decoder initial states."""
        batch = h_enc.shape[1]

        # Reshape: (num_layers, num_directions, batch, hidden) -> merge directions
        h = h_enc.view(self.num_layers, self.enc_directions, batch, self.hidden_dim)
        c = c_enc.view(self.num_layers, self.enc_directions, batch, self.hidden_dim)

        # Concatenate forward and backward, then project down
        h = h.permute(0, 2, 1, 3).reshape(self.num_layers, batch, -1)
        c = c.permute(0, 2, 1, 3).reshape(self.num_layers, batch, -1)

        h = torch.tanh(self.h_projection(h))
        c = torch.tanh(self.c_projection(c))
        return h.contiguous(), c.contiguous()

    def forward(
        self,
        x: torch.Tensor,
        teacher_forcing_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)
        teacher_forcing_target : (batch, horizon), optional ground truth for training
        teacher_forcing_ratio : probability of using ground truth at each decoder step

        Returns
        -------
        (batch, horizon) -- predicted ankle angle trajectory
        """
        _, (h_enc, c_enc) = self.encoder(x)
        h_dec, c_dec = self._bridge_hidden(h_enc, c_enc)

        batch = x.shape[0]
        # Start token: zero
        dec_input = torch.zeros(batch, 1, 1, device=x.device)

        outputs = []
        for t in range(self.horizon):
            dec_out, (h_dec, c_dec) = self.decoder(dec_input, (h_dec, c_dec))
            pred = self.output_layer(self.dropout(dec_out.squeeze(1)))  # (batch, 1)
            outputs.append(pred)

            if (
                teacher_forcing_target is not None
                and torch.rand(1).item() < teacher_forcing_ratio
            ):
                dec_input = teacher_forcing_target[:, t].unsqueeze(-1).unsqueeze(-1)
            else:
                dec_input = pred.unsqueeze(1)

        return torch.cat(outputs, dim=-1)  # (batch, horizon)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AnkleTransformer(nn.Module):
    """
    Self-attention encoder for Shadow-Limb prediction.

    Architecture:
        IMU window (B, T, C_in)
          -> Linear projection to d_model
          -> + learned positional embedding (T, d_model)
          -> N × TransformerEncoderLayer (self-attention + FFN)
          -> Pool on the **last** timestep (most recent sample)
          -> Linear head -> scalar ankle angle

    Why the last-step pool? For real-time control the most information-rich
    query is "what is the ankle angle right now?", i.e. the query at the
    most recent timestep. Mean-pooling over the window also works but
    blurs the contrast between stance and swing phases.
    """

    def __init__(
        self,
        input_dim: int = config.NUM_INPUT_CHANNELS,
        d_model: int = config.TRANSFORMER_D_MODEL,
        nhead: int = config.TRANSFORMER_NHEAD,
        num_layers: int = config.TRANSFORMER_NUM_LAYERS,
        dim_feedforward: int = config.TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = config.TRANSFORMER_DROPOUT,
        window_samples: int = config.WINDOW_SAMPLES,
        output_dim: int = 1,
        normalize_output: bool = False,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by nhead ({nhead})"
            )
        self.output_dim = output_dim
        self.normalize_output = normalize_output

        self.input_proj = nn.Linear(input_dim, d_model)

        # Learned positional embedding — with only 40 timesteps this is
        # cheaper and more expressive than the classical sinusoidal version
        # and lets the model specialize to the gait-window structure.
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, window_samples, d_model)
        )
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-norm: better stability at shallow depth
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        # Xavier init for the head/projection improves early-epoch stability
        # vs. PyTorch's default for Linear layers.
        for m in [self.input_proj, *self.head.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        h = self.input_proj(x) * math.sqrt(self.input_proj.out_features)
        h = h + self.pos_embedding[:, :seq_len]
        h = self.encoder(h)
        h = self.norm(h[:, -1])  # last timestep
        out = self.head(h)
        if self.normalize_output:
            out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            return out
        if self.output_dim == 1:
            return out.squeeze(-1)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(mode: str = config.PREDICTION_MODE, **kwargs) -> nn.Module:
    """
    Factory function to build the appropriate model variant.

    Direct-angle modes (scalar output, MSE on degrees):
        mode="single"       -> AnkleLSTM            (BiLSTM, next-step)
        mode="trajectory"   -> AnkleEncoderDecoder  (BiLSTM enc + LSTM dec)
        mode="transformer"  -> AnkleTransformer     (self-attention, next-step)

    Phase modes (2-D unit-norm output, MSE on (sin 2πφ, cos 2πφ), ankle
    angle recovered via reference-gait lookup at inference time):
        mode="phase_lstm"         -> AnkleLSTM(output_dim=2, normalize=True)
        mode="phase_transformer"  -> AnkleTransformer(output_dim=2, normalize=True)
    """
    if mode == "single":
        return AnkleLSTM(**kwargs)
    elif mode == "trajectory":
        return AnkleEncoderDecoder(**kwargs)
    elif mode == "transformer":
        return AnkleTransformer(**kwargs)
    elif mode == "phase_lstm":
        return AnkleLSTM(output_dim=2, normalize_output=True, **kwargs)
    elif mode == "phase_transformer":
        return AnkleTransformer(output_dim=2, normalize_output=True, **kwargs)
    else:
        raise ValueError(
            f"Unknown prediction mode: {mode!r}. "
            "Use 'single', 'trajectory', 'transformer', "
            "'phase_lstm', or 'phase_transformer'."
        )
