"""GRU and Transformer definitions shared by training and a future backend."""
import torch
from torch import nn


class ActivityNetwork(nn.Module):
    def __init__(self, kind, nfeatures, history_hours=168, horizon_hours=384,
                 hidden_size=32, dropout=.1):
        super().__init__()
        if kind not in {'gru', 'transformer'}:
            raise ValueError('kind must be gru or transformer')
        if hidden_size % 4 or min(history_hours, horizon_hours, nfeatures) < 1:
            raise ValueError('hidden_size must divide into four heads; dimensions must be positive')
        self.config = dict(kind=kind, nfeatures=nfeatures, history_hours=history_hours,
                           horizon_hours=horizon_hours, hidden_size=hidden_size, dropout=dropout)
        self.kind, self.history_hours, self.horizon_hours = kind, history_hours, horizon_hours
        if kind == 'gru':
            self.encoder = nn.GRU(nfeatures+1, hidden_size, batch_first=True)
        else:
            self.project = nn.Linear(nfeatures+1, hidden_size)
            self.position = nn.Parameter(torch.zeros(1, history_hours, hidden_size))
            nn.init.normal_(self.position, std=.02)
            self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(
                hidden_size, 4, hidden_size*2, dropout=dropout, batch_first=True), 1)
        self.head = nn.Sequential(nn.Linear(hidden_size+nfeatures+2, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, future):
        length = future.shape[1]
        if history.shape[1] != self.history_hours or not 1 <= length <= self.horizon_hours:
            raise ValueError('Wrong history length or future horizon exceeds training horizon.')
        if self.kind == 'gru':
            _, state = self.encoder(history)
            context = state[-1]
        else:
            context = self.encoder(self.project(history)+self.position).mean(dim=1)
        context = context[:, None, :].expand(-1, length, -1)
        lead = torch.arange(1, length+1, device=history.device, dtype=history.dtype)/self.horizon_hours
        lead = lead[None, :, None].expand(len(history), -1, -1)
        correction = self.head(torch.cat([context, future, lead], dim=-1)).squeeze(-1)
        return future[:, :, -1] + correction
