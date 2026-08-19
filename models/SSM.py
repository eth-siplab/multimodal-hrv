"""
Time-varying AR-2 Kalman (SSM) head for refining UNet IBI into HRV-grade RR.

State per step t:
    x_t = [RR_t, RR_{t-1}]^T

Dynamics (AR-2):
    RR_t = α_t * RR_{t-1} + β_t * RR_{t-2} + w_t
    
    x_t = A_t x_{t-1} + w_t
    A_t = [[α_t, β_t],
           [1,   0  ]]
    Q_t = [[q_t, 0],
           [0,   0]]  # Process noise only affects the new beat

Observation:
    m_t = [1, 0] x_t + δ_pat_t + v_t
    v_t ~ N(0, R_t)

Outputs:
    rr_hat    : [B, T]   filtered RR
"""

# ssm_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SSMHeadAR2(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 64, use_bias: bool = True, learn_R_scale: bool = True):
        super().__init__()
        self.use_bias = use_bias
        
        # Neural Network g(features) -> parameters
        self.g = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )

        # Heads for AR-2 Parameters
        self.head_alpha = nn.Linear(hidden, 1)   # alpha in (0, 1) -> damping/stability
        self.head_beta  = nn.Linear(hidden, 1)   # beta in (-1, 1) -> oscillation
        self.head_q     = nn.Linear(hidden, 1)   # q_t >= 0 -> process noise
        
        # Hemodynamic Offset (Delta PAT) - This acts as the "bias"
        if self.use_bias:
            self.head_dpat = nn.Linear(hidden, 1) 
        else:
            self.head_dpat = None

        # Optional global recalibration of R
        if learn_R_scale:
            self.log_R_scale = nn.Parameter(torch.tensor(math.log(1.0))) # Start at 1.0
        else:
            self.log_R_scale = None

        # Priors (ms scale)
        self.rr0 = nn.Parameter(torch.tensor(800.0))       # Initial HR guess (~75 bpm)
        self.P0  = nn.Parameter(torch.eye(2) * 100.0**2)   # Initial covariance
        self.eps = 1e-6

    def forward(
        self,
        m: torch.Tensor,                 # [B, T]  UNet IBI (measurement)
        logvar: torch.Tensor,            # [B, T]  UNet log-variance
        feats: torch.Tensor,             # [B, T, F] per-step features
        mask: torch.Tensor = None,       # [B, T] 1/0 valid mask
        warmup_w: float = 0.0
    ):
        B, T = m.shape
        dev = m.device
        if mask is None: mask = torch.ones_like(m)

        # --- 1. Predict Time-Varying Parameters ---
        h = self.g(feats)  # [B, T, H]

        # Alpha: (0, 1)
        raw_alpha = self.head_alpha(h).squeeze(-1)
        alpha = torch.sigmoid(raw_alpha)
        
        # Beta: (-1, 1)
        raw_beta = self.head_beta(h).squeeze(-1)
        beta = torch.tanh(raw_beta)

        # Q (Process Noise): Softplus
        q = F.softplus(self.head_q(h)).squeeze(-1) + self.eps

        # Hemodynamic Offset (Delta PAT)
        if self.use_bias:
            dpat = self.head_dpat(h).squeeze(-1)
        else:
            dpat = torch.zeros_like(m)

        # Observation Noise R
        finite_lv = torch.isfinite(logvar)
        R = torch.full_like(logvar, 1e8) 
        R[finite_lv] = torch.exp(logvar[finite_lv])
        
        if self.log_R_scale is not None:
            R = R * torch.exp(self.log_R_scale)
        R = torch.clamp(R, min=1.0) 

        # --- 2. Initialize State ---
        x_prev = torch.stack([self.rr0.expand(B), self.rr0.expand(B)], dim=1) # [B, 2]
        P_prev = self.P0.expand(B, 2, 2).clone()                              # [B, 2, 2]

        # Constants
        I = torch.eye(2, device=dev).unsqueeze(0).expand(B, 2, 2)             # [B, 2, 2]

        # Storage
        rr_hat = torch.zeros(B, T, device=dev)
        innovs = torch.zeros(B, T, device=dev)
        Ss     = torch.zeros(B, T, device=dev)

        # --- 3. Kalman Loop ---
        for t in range(T):
            # A_t Construction: [[alpha, beta], [1, 0]]
            A_t = torch.zeros(B, 2, 2, device=dev)
            A_t[:, 0, 0] = alpha[:, t]
            A_t[:, 0, 1] = beta[:, t]
            A_t[:, 1, 0] = 1.0
            
            # Q_t Construction
            Q_t = torch.zeros(B, 2, 2, device=dev)
            Q_t[:, 0, 0] = q[:, t]

            # Predict
            x_pred = torch.bmm(A_t, x_prev.unsqueeze(2)).squeeze(2) # [B, 2]
            P_pred = torch.bmm(torch.bmm(A_t, P_prev), A_t.transpose(1, 2)) + Q_t

            # Observe
            y_pred = x_pred[:, 0] + dpat[:, t]
            m_t = m[:, t]
            innov = m_t - y_pred
            S = P_pred[:, 0, 0] + R[:, t] + self.eps

            # Update
            K = P_pred[:, :, 0] / S.unsqueeze(1) # [B, 2]
            meas_ok = torch.isfinite(m_t) & (mask[:, t] > 0)
            
            innov_masked = torch.where(meas_ok, innov, torch.zeros_like(innov))
            x_new = x_pred + K * innov_masked.unsqueeze(1)
            
            KH = torch.zeros_like(P_prev)
            KH[:, :, 0] = K
            P_new = torch.bmm(I - KH, P_pred)
            P_new = torch.where(meas_ok.view(B, 1, 1), P_new, P_pred)

            # Store
            rr_hat[:, t] = x_new[:, 0]
            innovs[:, t] = innov
            Ss[:, t]     = S
            
            x_prev = x_new
            P_prev = P_new

        extras = {
            "innov": innovs, "S": Ss,
            "alpha": alpha, "beta": beta, "q": q, "dpat": dpat,
            "R": R
        }
        
        return rr_hat, dpat, extras

class CausalTransformerHead(nn.Module):
    """
    Ablation head: 2-layer causal Transformer.
    Takes the same inputs as SSMHeadAR2:
      m:      [B,T]
      logvar: [B,T]
      feats:  [B,T,F]
      mask:   [B,T] (1 valid, 0 invalid)

    Produces:
      rr_hat: [B,T]
      dpat:   [B,T] (interpretable as an additive correction; not physiological PAT)
      extras: dict
    """
    def __init__(
        self,
        feat_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        # NEW: chunked/windowed inference to avoid O(T^2) OOM on long subjects
        ctx_len: int = 1024,          # left context length
        block_len: int = 256,         # how many new steps per block
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.in_dim = feat_dim + 2  # m + logvar + feats

        self.ctx_len = int(ctx_len)
        self.block_len = int(block_len)

        self.in_proj = nn.Linear(self.in_dim, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.out_proj.bias)

    def _causal_mask_bool(self, T: int, device: torch.device) -> torch.Tensor:
        # Bool mask: True means "disallow attention"
        return torch.triu(torch.ones((T, T), dtype=torch.bool, device=device), diagonal=1)

    def _run_encoder(self, x_blk: torch.Tensor, key_padding_blk: torch.Tensor) -> torch.Tensor:
        # x_blk: [B,L,D], key_padding_blk: [B,L] bool
        L = x_blk.shape[1]
        attn_mask = self._causal_mask_bool(L, x_blk.device)  # [L,L] bool
        return self.encoder(x_blk, mask=attn_mask, src_key_padding_mask=key_padding_blk)

    def forward(
        self,
        m: torch.Tensor,                 # [B,T]
        logvar: torch.Tensor,            # [B,T]
        feats: torch.Tensor,             # [B,T,F]
        mask: torch.Tensor = None,       # [B,T]
        warmup_w: float = 0.0,           # unused, kept for call-compatibility
    ):
        B, T = m.shape
        dev = m.device
        if mask is None:
            mask = torch.ones_like(m)

        # sanitize numeric issues
        m_clean = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        lv_clean = torch.where(torch.isfinite(logvar), logvar, torch.zeros_like(logvar))
        feats_clean = torch.where(torch.isfinite(feats), feats, torch.zeros_like(feats))

        x = torch.cat([m_clean.unsqueeze(-1), lv_clean.unsqueeze(-1), feats_clean], dim=-1)  # [B,T,F+2]
        x = self.in_proj(x)  # [B,T,d_model]

        key_padding = (mask <= 0)  # [B,T] bool (True => pad/ignore)

        # --- chunked/windowed causal transformer ---
        delta = torch.empty((B, T), device=dev, dtype=x.dtype)

        if T <= (self.ctx_len + self.block_len):
            h = self._run_encoder(x, key_padding)
            delta[:] = self.out_proj(h).squeeze(-1)
        else:
            step = max(1, self.block_len)
            ctx = max(0, self.ctx_len)

            for t0 in range(0, T, step):
                t1 = min(T, t0 + step)
                left = max(0, t0 - ctx)

                x_blk = x[:, left:t1, :]                 # [B,L,D]
                kp_blk = key_padding[:, left:t1]         # [B,L]

                h_blk = self._run_encoder(x_blk, kp_blk) # [B,L,D]
                # keep only the newly-predicted region (last t1-t0 positions)
                h_new = h_blk[:, - (t1 - t0):, :]        # [B,step,D]
                delta[:, t0:t1] = self.out_proj(h_new).squeeze(-1)

        rr_hat = m_clean + delta
        rr_hat = torch.where(mask > 0, rr_hat, m_clean)

        extras = {"delta": delta}
        dpat = delta
        return rr_hat, dpat, extras